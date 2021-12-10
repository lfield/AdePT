// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "example13.h"
#include "example13.dp.h"

#include <AdePT/Atomic.h>
//#include <AdePT/BVHNavigator.h>
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

class MParray;
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
                              const vecgeom::VPlacedVolume *world, GlobalScoring *globalScoring, bool rotatingParticleGun, sycl::nd_item<3> item_ct1)
     
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
    //BVHNavigator::LocatePointIn(world, track.pos, track.navState, true);
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

void example13(int numParticles, double energy, int batch, const int *MCIndex,
               ScoringPerVolume *scoringPerVolume, GlobalScoring *globalScoring, int numVolumes,
               int numPlaced, G4HepEmState *state, bool rotatingParticleGun, const vecgeom::VPlacedVolume *world)
{
  sycl::queue q_ct1(sycl::default_selector{});
  std::cout <<  "Running on "
	        << q_ct1.get_device().get_info<cl::sycl::info::device::name>()
            << std::endl;
  dpct::device_ext &dev_ct1 = dpct::get_current_device();

  // Copy state to device.
  auto state_dev = malloc_device<G4HepEmState>(sizeof(state), q_ct1);
  q_ct1.memcpy(state_dev, state, sizeof(state)).wait();

  // Transfer MC indices.
  auto MCIndex_dev = malloc_device<int>(sizeof(int) * numVolumes, q_ct1);
  q_ct1.memcpy(MCIndex_dev, MCIndex, sizeof(int) * numVolumes).wait();

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
    particles[i].tracks = malloc_device<Track>(sizeof(Track) * TracksSize, q_ct1);
    particles[i].slotManager = malloc_device<SlotManager>(sizeof(SlotManager) * ManagerSize, q_ct1);
    particles[i].queues.currentlyActive = malloc_device<adept::MParray>(sizeof(adept::MParray) * QueueSize, q_ct1);
    particles[i].queues.nextActive = malloc_device<adept::MParray>(sizeof(adept::MParray) * QueueSize, q_ct1);
    q_ct1.submit([&](sycl::handler &cgh) {
      auto queues = particles[i].queues;
       cgh.single_task<class Init>([=]() {
         InitParticleQueues(queues, Capacity);
      });
    });   
    particles[i].stream = dev_ct1.create_queue();
  }
  dev_ct1.queues_wait_and_throw();

  ParticleType &electrons = particles[ParticleType::Electron];
  ParticleType &positrons = particles[ParticleType::Positron];
  ParticleType &gammas    = particles[ParticleType::Gamma];

  // Create a stream to synchronize kernels of all particle types.
  sycl::queue *stream;
  stream = dev_ct1.create_queue();

  // Allocate memory to score charged track length and energy deposit per volume.
  auto chargedTrackLength = malloc_device<double>(sizeof(double) * numPlaced, q_ct1);
  q_ct1.memset(chargedTrackLength, 0, sizeof(double) * numPlaced).wait();
  
  auto energyDeposit = malloc_device<double>(sizeof(double) * numPlaced, q_ct1);
  q_ct1.memset(energyDeposit, 0, sizeof(double) * numPlaced).wait();

  // Allocate and initialize scoring and statistics.
  auto globalScoring_dev = malloc_device<GlobalScoring>(sizeof(GlobalScoring), q_ct1);
  q_ct1.memset(globalScoring_dev, 0, sizeof(GlobalScoring)).wait();
  
  auto scoringPerVolume_dev = malloc_device<ScoringPerVolume>(sizeof(ScoringPerVolume), q_ct1);
  scoringPerVolume_dev->chargedTrackLength = chargedTrackLength;
  scoringPerVolume_dev->energyDeposit = energyDeposit;
  q_ct1.memcpy(scoringPerVolume, scoringPerVolume_dev, sizeof(ScoringPerVolume)).wait();

  auto stats = malloc_host<Stats>(sizeof(Stats), q_ct1);
  auto stats_dev = malloc_device<Stats>(sizeof(Stats), q_ct1);
  
  // Allocate memory to hold a "vanilla" SlotManager to initialize for each batch.
  SlotManager slotManagerInit(Capacity);
  auto  slotManagerInit_dev = malloc_device<SlotManager>(sizeof(SlotManager), q_ct1);
  q_ct1.memcpy(slotManagerInit_dev, &slotManagerInit, sizeof(SlotManager)).wait();

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
      q_ct1.memcpy(particles[i].slotManager, slotManagerInit_dev, ManagerSize).wait();
    }

    // Initialize primary particles.
    constexpr int InitThreads = 32;
    int initBlocks            = (chunk + InitThreads - 1) / InitThreads;
    ParticleGenerator electronGenerator(electrons.tracks, electrons.slotManager, electrons.queues.currentlyActive);
    //auto world_dev = vecgeom::cxx::CudaManager::Instance().world_gpu();
    {
      q_ct1.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, initBlocks) *
                                  sycl::range<3>(1, 1, InitThreads),
                              sycl::range<3>(1, 1, InitThreads)),
            [=](sycl::nd_item<3> item_ct1) {
              InitPrimaries(electronGenerator, startEvent, chunk, energy, world,
                            globalScoring, rotatingParticleGun, item_ct1);
            });
      });
    } 
    dev_ct1.queues_wait_and_throw();

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


      electrons.stream->submit([&](sycl::handler &cgh) {
        Track *electronsTracks = electrons.tracks;
        adept::MParray *currentlyActive = electrons.queues.currentlyActive;
        adept::MParray *nextActive = electrons.queues.nextActive;
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, transportBlocks) *
                                  sycl::range<3>(1, 1, TransportThreads),
                              sycl::range<3>(1, 1, TransportThreads)),
            [=](sycl::nd_item<3> item_ct1) {
             /*TransportElectrons<true>(electronsTracks,
                                       currentlyActive,
                                       secondaries, 
                                       nextActive,
                                       globalScoring_dev,
                                       scoringPerVolume_dev,
                                       item_ct1);
            */
            });
             
      });
      
      electrons.event_ct1 = std::chrono::steady_clock::now();

      electrons.event.wait();
        
      }

      // *** POSITRONS ***
      int numPositrons = stats->inFlight[ParticleType::Positron];
      if (numPositrons > 0) {
        transportBlocks = (numPositrons + TransportThreads - 1) / TransportThreads;
        transportBlocks = std::min(transportBlocks, MaxBlocks);

	    positrons.stream->submit([&](sycl::handler &cgh) {
          Track *positronsTracks = positrons.tracks;
          adept::MParray *pCurrentlyActive = positrons.queues.currentlyActive;
          adept::MParray *pNextActive = positrons.queues.nextActive;
          cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, transportBlocks) *
                              sycl::range<3>(1, 1, TransportThreads),
                              sycl::range<3>(1, 1, TransportThreads)),
            [=](sycl::nd_item<3> item_ct1) {
             /*   TransportElectrons<false>(positronsTracks,
                                        pCurrentlyActive,
                                        secondaries,
                                        pNextActive,
                                        globalScoring_dev,
                                        scoringPerVolume_dev,
                                        item_ct1);
            */
	    });
      });
      positrons.event_ct1 = std::chrono::steady_clock::now();
      positrons.event.wait();
      }

      // *** GAMMAS ***
      int numGammas = stats->inFlight[ParticleType::Gamma];
      if (numGammas > 0) {
        transportBlocks = (numGammas + TransportThreads - 1) / TransportThreads;
        transportBlocks = std::min(transportBlocks, MaxBlocks);
        gammas.stream->submit([&](sycl::handler &cgh) {
        Track *gammasTracks = gammas.tracks;
        adept::MParray *gCurrentlyActive = gammas.queues.currentlyActive;
        adept::MParray *gNextActive = gammas.queues.nextActive;
        cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, transportBlocks) *
                            sycl::range<3>(1, 1, TransportThreads),
                            sycl::range<3>(1, 1, TransportThreads)),
           [=](sycl::nd_item<3> item_ct1) {
           /*   TransportGammas(gammasTracks,
                              gCurrentlyActive,
                              secondaries,
                              gNextActive,
                              globalScoring_dev,
                              scoringPerVolume_dev,
                              item_ct1);
            */
            });
      });
      gammas.event_ct1 = std::chrono::steady_clock::now();
      gammas.event.wait();
      }

      // *** END OF TRANSPORT ***

      // The events ensure synchronization before finishing this iteration and
      // copying the Stats back to the host.
      AllParticleQueues queues = {{electrons.queues, positrons.queues, gammas.queues}};
      q_ct1.submit([&](sycl::handler &cgh) {
        cgh.single_task<class Finish>([=]() {
          FinishIteration(queues, stats_dev);
        });
      });
      q_ct1.memcpy(stats, stats_dev, sizeof(Stats));

      // Finally synchronize all kernels.
      q_ct1.wait();

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
      }
     }
  }
  std::cout << "done!" << std::endl;

  auto time = timer.Stop();
  std::cout << "Run time: " << time << "\n";

  // Transfer back scoring.
  q_ct1.memcpy(globalScoring, globalScoring_dev, sizeof(GlobalScoring)).wait();

  // Transfer back the scoring per volume (charged track length and energy deposit).
  q_ct1.memcpy(&scoringPerVolume, &scoringPerVolume_dev, sizeof(ScoringPerVolume)).wait();

  // Free resources.
  sycl::free(MCIndex_dev, q_ct1);
  sycl::free(chargedTrackLength, q_ct1);
  sycl::free(energyDeposit, q_ct1);
  sycl::free(globalScoring, q_ct1);
  sycl::free(&scoringPerVolume, q_ct1);
  sycl::free(stats_dev, q_ct1);
  sycl::free(stats, q_ct1);
  sycl::free(slotManagerInit_dev, q_ct1);
  dev_ct1.destroy_queue(stream);

  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    sycl::free(particles[i].tracks, q_ct1);
    sycl::free(particles[i].slotManager, q_ct1);
    sycl::free(particles[i].queues.currentlyActive, q_ct1);
    sycl::free(particles[i].queues.nextActive, q_ct1);
    dev_ct1.destroy_queue(particles[i].stream);
  }
}
