// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>
#include "example13.h"
#include "example13-cu.h"

#include <AdePT/Atomic.h>
#include <AdePT/BVHNavigator.h>
#include <AdePT/MParray.h>

#include <CopCore/Global.h>
#include <CopCore/PhysicalConstants.h>
#include <CopCore/Ranluxpp.h>

#include <VecGeom/base/Config.h>
#include <VecGeom/base/Stopwatch.h>
#include <VecGeom/management/GeoManager.h>
#ifdef VECGEOM_ENABLE_CUDA
#include <VecGeom/backend/cuda/Interface.h>
#endif

#include <G4HepEmState.hh>
#include <G4HepEmData.hh>
#include <G4HepEmElectronInit.hh>
#include <G4HepEmGammaInit.hh>
#include <G4HepEmMatCutData.hh>
#include <G4HepEmMaterialInit.hh>
#include <G4HepEmParameters.hh>
#include <G4HepEmParametersInit.hh>

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <chrono>

//dpct::constant_memory<struct G4HepEmParameters, 0> g4HepEmPars;
//dpct::constant_memory<struct G4HepEmData, 0> g4HepEmData;

//dpct::constant_memory<int *, 0> MCIndex;

//dpct::constant_memory<int, 0> Zero(0);

void InitG4HepEmGPU(G4HepEmState *state)
{
  // Copy to GPU.
  //CopyG4HepEmDataToGPU(state->fData);
  //COPCORE_CUDA_CHECK(cudaMemcpyToSymbol(g4HepEmPars, state->fParameters, sizeof(G4HepEmParameters)));
  /*
  // Create G4HepEmData with the device pointers.
  G4HepEmData dataOnDevice;
  dataOnDevice.fTheMatCutData   = state->fData->fTheMatCutData_gpu;
  dataOnDevice.fTheMaterialData = state->fData->fTheMaterialData_gpu;
  dataOnDevice.fTheElementData  = state->fData->fTheElementData_gpu;
  dataOnDevice.fTheElectronData = state->fData->fTheElectronData_gpu;
  dataOnDevice.fThePositronData = state->fData->fThePositronData_gpu;
  dataOnDevice.fTheSBTableData  = state->fData->fTheSBTableData_gpu;
  dataOnDevice.fTheGSTableData  = state->fData->fTheGSTableData_gpu;
  dataOnDevice.fTheGammaData    = state->fData->fTheGammaData_gpu;
  // The other pointers should never be used.
  dataOnDevice.fTheMatCutData_gpu   = nullptr;
  dataOnDevice.fTheMaterialData_gpu = nullptr;
  dataOnDevice.fTheElementData_gpu  = nullptr;
  dataOnDevice.fTheElectronData_gpu = nullptr;
  dataOnDevice.fThePositronData_gpu = nullptr;
  dataOnDevice.fTheSBTableData_gpu  = nullptr;
  dataOnDevice.fTheGSTableData_gpu  = nullptr;
  dataOnDevice.fTheGammaData_gpu    = nullptr;

  COPCORE_CUDA_CHECK(cudaMemcpyToSymbol(g4HepEmData, &dataOnDevice, sizeof(G4HepEmData)));
  */
}

// A bundle of queues per particle type:
//  * Two for active particles, one for the current iteration and the second for the next.
struct ParticleQueues {
  adept::MParray *currentlyActive;
  adept::MParray *nextActive;

  void SwapActive() { std::swap(currentlyActive, nextActive); }
};

struct ParticleType {
  Track *tracks;
  SlotManager *slotManager;
  ParticleQueues queues;
  sycl::queue *stream;
  sycl::event event;
  std::chrono::time_point<std::chrono::steady_clock> event_ct1;

  enum {
    Electron = 0,
    Positron = 1,
    Gamma    = 2,

    NumParticleTypes,
  };
};

// A bundle of queues for the three particle types.
struct AllParticleQueues {
  ParticleQueues queues[ParticleType::NumParticleTypes];
};

// Kernel to initialize the set of queues per particle type.
void InitParticleQueues(ParticleQueues queues, size_t Capacity)
{
  adept::MParray::MakeInstanceAt(Capacity, queues.currentlyActive);
  adept::MParray::MakeInstanceAt(Capacity, queues.nextActive);
}

// Kernel function to initialize a set of primary particles.
void InitPrimaries(ParticleGenerator generator, int startEvent, int numEvents, double energy,
                              const vecgeom::VPlacedVolume *world, GlobalScoring *globalScoring, bool rotatingParticleGun,
                              sycl::nd_item<3> item_ct1)
{
  for (int i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
               item_ct1.get_local_id(2);
       i < numEvents;
       i += item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2)) {
    Track &track = generator.NextTrack();

    track.rngState.SetSeed(startEvent + i);
    track.energy       = energy;
    track.numIALeft[0] = -1.0;
    track.numIALeft[1] = -1.0;
    track.numIALeft[2] = -1.0;
    track.initialRange = -1.0;

    track.pos = {0, 0, 0};
    if (rotatingParticleGun) {
      // Generate particles flat in phi and in eta between -5 and 5. We'll lose the far forwards ones, so no need to simulate.
      const double phi = 2. * M_PI * track.rngState.Rndm();
      const double eta = -5. + 10. * track.rngState.Rndm();
      track.dir = {
        cos(phi) / cosh(eta),
        sin(phi) / cosh(eta),
                   tanh(eta)};
    } else {
      track.dir = {1.0, 0, 0};
    }
    track.navState.Clear();
    BVHNavigator::LocatePointIn(world, track.pos, track.navState, true);

    sycl::atomic<unsigned long long>(
        sycl::global_ptr<unsigned long long>(&globalScoring->numElectrons))
        .fetch_add(1);
  }
}

// A data structure to transfer statistics after each iteration.
struct Stats {
  int inFlight[ParticleType::NumParticleTypes];
};

// Finish iteration: clear queues and fill statistics.
void FinishIteration(AllParticleQueues all, Stats *stats)
{
  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    all.queues[i].currentlyActive->clear();
    stats->inFlight[i] = all.queues[i].nextActive->size();
  }
}

void ClearQueue(adept::MParray *queue)
{
  queue->clear();
}

void example13(int numParticles, double energy, int batch, const int *MCIndex_host,
               ScoringPerVolume *scoringPerVolume_host, GlobalScoring *globalScoring_host, int numVolumes,
               int numPlaced, G4HepEmState *state, bool rotatingParticleGun)
{
  sycl::queue q_ct1(sycl::default_selector{});
  //InitG4HepEmGPU(state);

  // Transfer MC indices.
  int *MCIndex_dev = nullptr;
  //  COPCORE_CUDA_CHECK(cudaMalloc(&MCIndex_dev, sizeof(int) * numVolumes));
  /*
  DPCT1003:1: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  //COPCORE_CUDA_CHECK(
  //   (q_ct1.memcpy(MCIndex_dev, MCIndex_host, sizeof(int) * numVolumes).wait(),
  //   0));
  /*
  DPCT1003:2: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  //COPCORE_CUDA_CHECK(
  //   (q_ct1.memcpy(MCIndex.get_ptr(), &MCIndex_dev, sizeof(int *)).wait(), 0));

  // Capacity of the different containers aka the maximum number of particles.
  constexpr int Capacity = 256 * 1024;

  std::cout << "INFO: capacity of containers set to " << Capacity << std::endl;
  if (batch == -1) {
    // Rule of thumb: at most 2000 particles of one type per GeV primary.
    batch = Capacity / ((int)energy / copcore::units::GeV) / 2000;
  } else if (batch < 1) {
    batch = 1;
  }
  std::cout << "INFO: batching " << batch << " particles for transport on the GPU" << std::endl;
  if (BzFieldValue != 0) {
    std::cout << "INFO: running with field Bz = " << BzFieldValue / copcore::units::tesla << " T" << std::endl;
  } else {
    std::cout << "INFO: running with magnetic field OFF" << std::endl;
  }

  // Allocate structures to manage tracks of an implicit type:
  //  * memory to hold the actual Track elements,
  //  * objects to manage slots inside the memory,
  //  * queues of slots to remember active particle and those needing relocation,
  //  * a stream and an event for synchronization of kernels.
  constexpr size_t TracksSize  = sizeof(Track) * Capacity;
  constexpr size_t ManagerSize = sizeof(SlotManager);
  const size_t QueueSize       = adept::MParray::SizeOfInstance(Capacity);

  ParticleType particles[ParticleType::NumParticleTypes];
  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    // COPCORE_CUDA_CHECK(cudaMalloc(&particles[i].tracks, TracksSize));

    //COPCORE_CUDA_CHECK(cudaMalloc(&particles[i].slotManager, ManagerSize));

    //COPCORE_CUDA_CHECK(cudaMalloc(&particles[i].queues.currentlyActive, QueueSize));
    //COPCORE_CUDA_CHECK(cudaMalloc(&particles[i].queues.nextActive, QueueSize));
    //InitParticleQueues<<<1, 1>>>(particles[i].queues, Capacity);

    /*
    DPCT1003:3: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    // COPCORE_CUDA_CHECK((particles[i].stream = dev_ct1.create_queue(), 0));
    /*
    DPCT1027:4: The call to cudaEventCreate was replaced with 0 because this
    call is redundant in DPC++.
    */
    COPCORE_CUDA_CHECK(0);
  }
  /*
  DPCT1003:5: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  //COPCORE_CUDA_CHECK((dev_ct1.queues_wait_and_throw(), 0));

  ParticleType &electrons = particles[ParticleType::Electron];
  ParticleType &positrons = particles[ParticleType::Positron];
  ParticleType &gammas    = particles[ParticleType::Gamma];

  // Create a stream to synchronize kernels of all particle types.
  sycl::queue *stream;
  /*
  DPCT1003:6: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  //COPCORE_CUDA_CHECK((stream = dev_ct1.create_queue(), 0));

  // Allocate memory to score charged track length and energy deposit per volume.
  double *chargedTrackLength = nullptr;
  //  COPCORE_CUDA_CHECK(cudaMalloc(&chargedTrackLength, sizeof(double) * numPlaced));
  /*
  DPCT1003:7: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  //COPCORE_CUDA_CHECK(
  //    (q_ct1.memset(chargedTrackLength, 0, sizeof(double) * numPlaced).wait(),
  //     0));
  double *energyDeposit = nullptr;
  //  COPCORE_CUDA_CHECK(cudaMalloc(&energyDeposit, sizeof(double) * numPlaced));
  /*
  DPCT1003:8: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  // COPCORE_CUDA_CHECK(
  //    (q_ct1.memset(energyDeposit, 0, sizeof(double) * numPlaced).wait(), 0));

  // Allocate and initialize scoring and statistics.
  GlobalScoring *globalScoring = nullptr;
  //COPCORE_CUDA_CHECK(cudaMalloc(&globalScoring, sizeof(GlobalScoring)));
  /*
  DPCT1003:9: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  //COPCORE_CUDA_CHECK(
  //  (q_ct1.memset(globalScoring, 0, sizeof(GlobalScoring)).wait(), 0));

  ScoringPerVolume *scoringPerVolume = nullptr;
  ScoringPerVolume scoringPerVolume_devPtrs;
  scoringPerVolume_devPtrs.chargedTrackLength = chargedTrackLength;
  scoringPerVolume_devPtrs.energyDeposit      = energyDeposit;
  // COPCORE_CUDA_CHECK(cudaMalloc(&scoringPerVolume, sizeof(ScoringPerVolume)));
  //COPCORE_CUDA_CHECK(
      /*
      DPCT1003:10: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
  //    (q_ct1
  //         .memcpy(scoringPerVolume, &scoringPerVolume_devPtrs,
  //                 sizeof(ScoringPerVolume))
  //         .wait(),
  //     0));

  Stats *stats_dev = nullptr;
  //COPCORE_CUDA_CHECK(cudaMalloc(&stats_dev, sizeof(Stats)));
  Stats *stats = nullptr;
  //COPCORE_CUDA_CHECK(cudaMallocHost(&stats, sizeof(Stats)));

  // Allocate memory to hold a "vanilla" SlotManager to initialize for each batch.
  SlotManager slotManagerInit(Capacity);
  SlotManager *slotManagerInit_dev = nullptr;
  //COPCORE_CUDA_CHECK(cudaMalloc(&slotManagerInit_dev, sizeof(SlotManager)));
  /*
  DPCT1003:11: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  //COPCORE_CUDA_CHECK(
  //   (q_ct1.memcpy(slotManagerInit_dev, &slotManagerInit, sizeof(SlotManager))
  //           .wait(),
  //    0));

  vecgeom::Stopwatch timer;
  timer.Start();

  std::cout << std::endl << "Simulating particles ";
  const bool detailed = (numParticles / batch) < 50;
  if (!detailed) {
    std::cout << "... " << std::flush;
  }

  for (int startEvent = 1; startEvent <= numParticles; startEvent += batch) {
    if (detailed) {
      std::cout << startEvent << " ... " << std::flush;
    }
    int left  = numParticles - startEvent + 1;
    int chunk = std::min(left, batch);

    for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
      //      COPCORE_CUDA_CHECK(cudaMemcpyAsync(particles[i].slotManager, slotManagerInit_dev, ManagerSize,
      //                                 cudaMemcpyDeviceToDevice, stream));
    }

    // Initialize primary particles.
    constexpr int InitThreads = 32;
    int initBlocks            = (chunk + InitThreads - 1) / InitThreads;
    ParticleGenerator electronGenerator(electrons.tracks, electrons.slotManager, electrons.queues.currentlyActive);
    //    auto world_dev = vecgeom::cxx::CudaManager::Instance().world_gpu();
    // InitPrimaries<<<initBlocks, InitThreads, 0, stream>>>(electronGenerator, startEvent, chunk, energy, world_dev,
    //                                                    globalScoring, rotatingParticleGun);
    /*
    DPCT1003:12: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    //COPCORE_CUDA_CHECK((stream->wait(), 0));

    stats->inFlight[ParticleType::Electron] = chunk;
    stats->inFlight[ParticleType::Positron] = 0;
    stats->inFlight[ParticleType::Gamma]    = 0;

    constexpr int MaxBlocks        = 1024;
    constexpr int TransportThreads = 32;
    int transportBlocks;

    int inFlight;
    int loopingNo         = 0;
    int previousElectrons = -1, previousPositrons = -1;

    do {
      Secondaries secondaries = {
          .electrons = {electrons.tracks, electrons.slotManager, electrons.queues.nextActive},
          .positrons = {positrons.tracks, positrons.slotManager, positrons.queues.nextActive},
          .gammas    = {gammas.tracks, gammas.slotManager, gammas.queues.nextActive},
      };

      // *** ELECTRONS ***
      int numElectrons = stats->inFlight[ParticleType::Electron];
      if (numElectrons > 0) {
        transportBlocks = (numElectrons + TransportThreads - 1) / TransportThreads;
        transportBlocks = std::min(transportBlocks, MaxBlocks);

	//        TransportElectrons<<<transportBlocks, TransportThreads, 0, electrons.stream>>>(
        //    electrons.tracks, electrons.queues.currentlyActive, secondaries, electrons.queues.nextActive, globalScoring,
	//   scoringPerVolume);

        /*
        DPCT1012:13: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:14: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        electrons.event_ct1 = std::chrono::steady_clock::now();
        COPCORE_CUDA_CHECK(
            (electrons.event = electrons.stream->submit_barrier(), 0));
        /*
        DPCT1003:15: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        COPCORE_CUDA_CHECK(
            (electrons.event = stream->submit_barrier({electrons.event}), 0));
      }

      // *** POSITRONS ***
      int numPositrons = stats->inFlight[ParticleType::Positron];
      if (numPositrons > 0) {
        transportBlocks = (numPositrons + TransportThreads - 1) / TransportThreads;
        transportBlocks = std::min(transportBlocks, MaxBlocks);

	//        TransportPositrons<<<transportBlocks, TransportThreads, 0, positrons.stream>>>(
        //    positrons.tracks, positrons.queues.currentlyActive, secondaries, positrons.queues.nextActive, globalScoring,
	//   scoringPerVolume);

        /*
        DPCT1012:16: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:17: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        positrons.event_ct1 = std::chrono::steady_clock::now();
        COPCORE_CUDA_CHECK(
            (positrons.event = positrons.stream->submit_barrier(), 0));
        /*
        DPCT1003:18: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        COPCORE_CUDA_CHECK(
            (positrons.event = stream->submit_barrier({positrons.event}), 0));
      }

      // *** GAMMAS ***
      int numGammas = stats->inFlight[ParticleType::Gamma];
      if (numGammas > 0) {
        transportBlocks = (numGammas + TransportThreads - 1) / TransportThreads;
        transportBlocks = std::min(transportBlocks, MaxBlocks);

	//        TransportGammas<<<transportBlocks, TransportThreads, 0, gammas.stream>>>(
        //    gammas.tracks, gammas.queues.currentlyActive, secondaries, gammas.queues.nextActive, globalScoring,
	//   scoringPerVolume);

        /*
        DPCT1012:19: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:20: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        gammas.event_ct1 = std::chrono::steady_clock::now();
        COPCORE_CUDA_CHECK((gammas.event = gammas.stream->submit_barrier(), 0));
        /*
        DPCT1003:21: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        COPCORE_CUDA_CHECK(
            (gammas.event = stream->submit_barrier({gammas.event}), 0));
      }

      // *** END OF TRANSPORT ***

      // The events ensure synchronization before finishing this iteration and
      // copying the Stats back to the host.
      AllParticleQueues queues = {{electrons.queues, positrons.queues, gammas.queues}};
      stream->parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            FinishIteration(queues, stats_dev);
          });
      /*
      DPCT1003:22: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      COPCORE_CUDA_CHECK((stream->memcpy(stats, stats_dev, sizeof(Stats)), 0));

      // Finally synchronize all kernels.
      /*
      DPCT1003:23: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      COPCORE_CUDA_CHECK((stream->wait(), 0));

      // Count the number of particles in flight.
      inFlight = 0;
      for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
        inFlight += stats->inFlight[i];
      }

      // Swap the queues for the next iteration.
      electrons.queues.SwapActive();
      positrons.queues.SwapActive();
      gammas.queues.SwapActive();

      // Check if only charged particles are left that are looping.
      numElectrons = stats->inFlight[ParticleType::Electron];
      numPositrons = stats->inFlight[ParticleType::Positron];
      numGammas    = stats->inFlight[ParticleType::Gamma];
      if (numElectrons == previousElectrons && numPositrons == previousPositrons && numGammas == 0) {
        loopingNo++;
      } else {
        previousElectrons = numElectrons;
        previousPositrons = numPositrons;
        loopingNo         = 0;
      }

    } while (inFlight > 0 && loopingNo < 200);

    if (inFlight > 0) {
      for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
        ParticleType &pType   = particles[i];
        int inFlightParticles = stats->inFlight[i];
        if (inFlightParticles == 0) {
          continue;
        }

	//  ClearQueue<<<1, 1, 0, stream>>>(pType.queues.currentlyActive);
      }
      /*
      DPCT1003:24: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      //COPCORE_CUDA_CHECK((stream->wait(), 0));
    }
  }
  std::cout << "done!" << std::endl;

  auto time = timer.Stop();
  std::cout << "Run time: " << time << "\n";

  // Transfer back scoring.
  /*
  DPCT1003:25: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK(
      (q_ct1.memcpy(globalScoring_host, globalScoring, sizeof(GlobalScoring))
           .wait(),
       0));

  // Transfer back the scoring per volume (charged track length and energy deposit).
  /*
  DPCT1003:26: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((q_ct1
                          .memcpy(scoringPerVolume_host->chargedTrackLength,
                                  scoringPerVolume_devPtrs.chargedTrackLength,
                                  sizeof(double) * numPlaced)
                          .wait(),
                      0));
  /*
  DPCT1003:27: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((q_ct1
                          .memcpy(scoringPerVolume_host->energyDeposit,
                                  scoringPerVolume_devPtrs.energyDeposit,
                                  sizeof(double) * numPlaced)
                          .wait(),
                      0));

  // Free resources.
  /*
  DPCT1003:28: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((sycl::free(MCIndex_dev, q_ct1), 0));
  /*
  DPCT1003:29: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((sycl::free(chargedTrackLength, q_ct1), 0));
  /*
  DPCT1003:30: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((sycl::free(energyDeposit, q_ct1), 0));

  /*
  DPCT1003:31: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((sycl::free(globalScoring, q_ct1), 0));
  /*
  DPCT1003:32: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((sycl::free(scoringPerVolume, q_ct1), 0));
  /*
  DPCT1003:33: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((sycl::free(stats_dev, q_ct1), 0));
  /*
  DPCT1003:34: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((sycl::free(stats, q_ct1), 0));
  /*
  DPCT1003:35: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  COPCORE_CUDA_CHECK((sycl::free(slotManagerInit_dev, q_ct1), 0));

  /*
  DPCT1003:36: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  //COPCORE_CUDA_CHECK((dev_ct1.destroy_queue(stream), 0));

  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    /*
    DPCT1003:37: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    COPCORE_CUDA_CHECK((sycl::free(particles[i].tracks, q_ct1), 0));
    /*
    DPCT1003:38: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    COPCORE_CUDA_CHECK((sycl::free(particles[i].slotManager, q_ct1), 0));

    /*
    DPCT1003:39: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    COPCORE_CUDA_CHECK(
        (sycl::free(particles[i].queues.currentlyActive, q_ct1), 0));
    /*
    DPCT1003:40: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    COPCORE_CUDA_CHECK((sycl::free(particles[i].queues.nextActive, q_ct1), 0));

    /*
    DPCT1003:41: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    //COPCORE_CUDA_CHECK((dev_ct1.destroy_queue(particles[i].stream), 0));
    /*
    DPCT1027:42: The call to cudaEventDestroy was replaced with 0 because this
    call is redundant in DPC++.
    */
    COPCORE_CUDA_CHECK(0);
  }
}
