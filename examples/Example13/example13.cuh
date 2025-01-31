// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef EXAMPLE13_CUH
#define EXAMPLE13_CUH

#include <CL/sycl.hpp>
#include "example13.h"

#include <AdePT/MParray.h>
#include <CopCore/SystemOfUnits.h>
#include <CopCore/Ranluxpp.h>

#include <G4HepEmData.hh>
#include <G4HepEmParameters.hh>
#include <G4HepEmRandomEngine.hh>

#include <VecGeom/base/Vector3D.h>
#include <VecGeom/navigation/NavStateIndex.h>

// A data structure to represent a particle track. The particle type is implicit
// by the queue and not stored in memory.
struct Track {
  using Precision = vecgeom::Precision;
  RanluxppDouble rngState;
  double energy;
  double numIALeft[3];
  double initialRange;

  vecgeom::Vector3D<Precision> pos;
  vecgeom::Vector3D<Precision> dir;
  vecgeom::NavStateIndex navState;

  double Uniform() { return rngState.Rndm(); }

  void InitAsSecondary(const Track &parent)
  {
    // The caller is responsible to branch a new RNG state and to set the energy.
    this->numIALeft[0] = -1.0;
    this->numIALeft[1] = -1.0;
    this->numIALeft[2] = -1.0;
    this->initialRange = -1.0;

    // A secondary inherits the position of its parent; the caller is responsible
    // to update the directions.
    this->pos      = parent.pos;
    this->navState = parent.navState;
  }
};

// Defined in example13.cu
//dpct::constant_memory<int, 0> Zero;
int Zero = 0;

class RanluxppDoubleEngine : public G4HepEmRandomEngine {
  // Wrapper functions to call into RanluxppDouble.
  static __attribute__((noinline)) double FlatWrapper(void *object)
  {
    return ((RanluxppDouble *)object)->Rndm();
  }
  static __attribute__((noinline)) void FlatArrayWrapper(void *object, const int size, double *vect)
  {
    for (int i = 0; i < size; i++) {
      vect[i] = ((RanluxppDouble *)object)->Rndm();
    }
  }

public:
  RanluxppDoubleEngine(RanluxppDouble *engine)
      : G4HepEmRandomEngine(/*object=*/engine, &FlatWrapper, &FlatArrayWrapper)
  {
#ifdef DPCT_COMPATIBILITY_TEMP
    // This is a hack: The compiler cannot see that we're going to call the
    // functions through their pointers, so it underestimates the number of
    // required registers. By including calls to the (non-inlinable) functions
    // we force the compiler to account for the register usage, even if this
    // particular set of calls are not executed at runtime.
    if (Zero) {
      FlatWrapper(engine);
      FlatArrayWrapper(engine, 0, nullptr);
    }
#endif
  }
};

// A data structure to manage slots in the track storage.
class SlotManager {
  adept::Atomic_t<int> fNextSlot;
  const int fMaxSlot;

public:
  SlotManager(int maxSlot) : fMaxSlot(maxSlot) { fNextSlot = 0; }

  int NextSlot()
  {
    int next = fNextSlot.fetch_add(1);
    if (next >= fMaxSlot) return -1;
    return next;
  }
};

// A bundle of pointers to generate particles of an implicit type.
class ParticleGenerator {
  Track *fTracks;
  SlotManager *fSlotManager;
  adept::MParray *fActiveQueue;

public:
  ParticleGenerator(Track *tracks, SlotManager *slotManager, adept::MParray *activeQueue)
      : fTracks(tracks), fSlotManager(slotManager), fActiveQueue(activeQueue)
  {
  }

  Track &NextTrack()
  {
    int slot = fSlotManager->NextSlot();
    if (slot == -1) {
      //COPCORE_EXCEPTION("No slot available in ParticleGenerator::NextTrack");
    }
    fActiveQueue->push_back(slot);
    return fTracks[slot];
  }
};

// A bundle of generators for the three particle types.
struct Secondaries {
  ParticleGenerator electrons;
  ParticleGenerator positrons;
  ParticleGenerator gammas;
};

// Kernels in different TUs.
template <bool IsElectron>
SYCL_EXTERNAL void TransportElectrons(Track *electrons, const adept::MParray *active, Secondaries secondaries,
                                   adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                   ScoringPerVolume *scoringPerVolume, sycl::nd_item<3> item_ct1);
SYCL_EXTERNAL void TransportPositrons(Track *positrons, const adept::MParray *active, Secondaries secondaries,
                                   adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                   ScoringPerVolume *scoringPerVolume, sycl::nd_item<3> item_ct1);

SYCL_EXTERNAL void TransportGammas(Track *gammas, const adept::MParray *active, Secondaries secondaries,
                                adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                ScoringPerVolume *scoringPerVolume, sycl::nd_item<3> item_ct1);

// Constant data structures from G4HepEm accessed by the kernels.
// (defined in TestEm3.cu)
G4HepEmParameters g4HepEmPars;
G4HepEmData g4HepEmData;

int * MCIndex;

// constexpr float BzFieldValue = 0.1 * copcore::units::tesla;
constexpr double BzFieldValue = 0;

#endif
