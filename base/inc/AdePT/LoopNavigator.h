// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file LoopNavigator.h
 * @brief Navigation methods for geometry.
 */

#ifndef RT_LOOP_NAVIGATOR_H_
#define RT_LOOP_NAVIGATOR_H_

#include <CopCore/Global.h>

#include <VecGeom/base/Global.h>
#include <VecGeom/base/Vector3D.h>
#include <VecGeom/navigation/NavStateIndex.h>

#ifdef VECGEOM_ENABLE_CUDA
#include <VecGeom/backend/cuda/Interface.h>
#endif

inline namespace COPCORE_IMPL {

class LoopNavigator {

public:
  using VPlacedVolumePtr_t = vecgeom::VPlacedVolume const *;

  __host__ __device__
  static VPlacedVolumePtr_t LocatePointIn(vecgeom::VPlacedVolume const *vol,
                                          vecgeom::Vector3D<vecgeom::Precision> const &point,
                                          vecgeom::NavStateIndex &path, bool top,
                                          vecgeom::VPlacedVolume const *exclude = nullptr)
  {
    if (top) {
      assert(vol != nullptr);
      if (!vol->UnplacedContains(point)) return nullptr;
    }

    VPlacedVolumePtr_t currentvolume = vol;
    vecgeom::Vector3D<vecgeom::Precision> currentpoint(point);
    path.Push(currentvolume);

    bool godeeper;
    do {
      godeeper = false;
      for (auto *daughter : currentvolume->GetDaughters()) {
        if (daughter == exclude) {
          continue;
        }
        vecgeom::Vector3D<vecgeom::Precision> transformedpoint;
        if (daughter->Contains(currentpoint, transformedpoint)) {
          path.Push(daughter);
          currentpoint  = transformedpoint;
          currentvolume = daughter;
          godeeper      = true;
          break;
        }
      }

      // Only exclude the placed volume once since we could enter it again via a
      // different volume history.
      exclude = nullptr;
    } while (godeeper);

    return currentvolume;
  }

  __host__ __device__
  static VPlacedVolumePtr_t RelocatePoint(vecgeom::Vector3D<vecgeom::Precision> const &localpoint,
                                          vecgeom::NavStateIndex &path)
  {
    vecgeom::VPlacedVolume const *currentmother       = path.Top();
    vecgeom::Vector3D<vecgeom::Precision> transformed = localpoint;
    do {
      path.Pop();
      transformed   = currentmother->GetTransformation()->InverseTransform(transformed);
      currentmother = path.Top();
    } while (currentmother && (currentmother->IsAssembly() || !currentmother->UnplacedContains(transformed)));

    if (currentmother) {
      path.Pop();
      return LocatePointIn(currentmother, transformed, path, false);
    }
    return currentmother;
  }

private:
  // Computes a step in the current volume from the localpoint into localdir,
  // taking step_limit into account. If a volume is hit, the function calls
  // out_state.SetBoundaryState(true) and hitcandidate is set to the hit
  // daughter volume, or kept unchanged if the current volume is left.
  __host__ __device__ static double ComputeStepAndHit(vecgeom::Vector3D<vecgeom::Precision> const &localpoint,
                                                      vecgeom::Vector3D<vecgeom::Precision> const &localdir,
                                                      vecgeom::Precision step_limit,
                                                      vecgeom::NavStateIndex const &in_state,
                                                      vecgeom::NavStateIndex &out_state,
                                                      VPlacedVolumePtr_t &hitcandidate)
  {
    vecgeom::Precision step         = step_limit;
    VPlacedVolumePtr_t pvol         = in_state.Top();

    // need to calc DistanceToOut first
    step = pvol->DistanceToOut(localpoint, localdir, step_limit);

    if (step < 0) step = 0;

    for (auto *daughter : pvol->GetDaughters()) {
      double ddistance = daughter->DistanceToIn(localpoint, localdir, step);

      // If ddistance is negative, we are already inside that daughter. The most
      // likely case is that we just left it and should should not enter again.
      const bool valid = (ddistance >= 0 && ddistance < step && !vecgeom::IsInf(ddistance));
      hitcandidate = valid ? daughter : hitcandidate;
      step         = valid ? ddistance : step;
    }

    // now we have the candidates and we prepare the out_state
    in_state.CopyTo(&out_state);
    if (step == vecgeom::kInfLength && step_limit > 0.) {
      out_state.SetBoundaryState(true);
      do {
        out_state.Pop();
      } while (out_state.Top()->IsAssembly());

      return vecgeom::kTolerance;
    }

    // Is geometry further away than physics step?
    if (step > step_limit) {
      // Then this is a phyics step and we don't need to do anything.
      out_state.SetBoundaryState(false);
      return step_limit;
    }

    // Otherwise it is a geometry step and we push the point to the boundary.
    out_state.SetBoundaryState(true);

    return step;
  }

public:
  // Computes the isotropic safety from the globalpoint.
  __host__ __device__ static double ComputeSafety(vecgeom::Vector3D<vecgeom::Precision> const &globalpoint,
                                                  vecgeom::NavStateIndex const &state)
  {
    VPlacedVolumePtr_t pvol = state.Top();
    vecgeom::Transformation3D m;
    state.TopMatrix(m);
    vecgeom::Vector3D<vecgeom::Precision> localpoint = m.Transform(globalpoint);

    vecgeom::Precision safety = pvol->SafetyToOut(localpoint);

    for (auto *daughter : pvol->GetDaughters()) {
      double dsafety = daughter->SafetyToIn(localpoint);
      safety = dsafety < safety ? dsafety : safety;
    }

    return safety;
  }

  // Computes a step from the globalpoint (which must be in the current volume)
  // into globaldir, taking step_limit into account. If a volume is hit, the
  // function calls out_state.SetBoundaryState(true) and relocates the state to
  // the next volume.
  __host__ __device__ static double ComputeStepAndPropagatedState(
      vecgeom::Vector3D<vecgeom::Precision> const &globalpoint, vecgeom::Vector3D<vecgeom::Precision> const &globaldir,
      vecgeom::Precision step_limit, vecgeom::NavStateIndex const &in_state, vecgeom::NavStateIndex &out_state, vecgeom::Precision push = 0.)
  {
    constexpr vecgeom::Precision kPush = 10. * vecgeom::kTolerance;
    // calculate local point/dir from global point/dir
    vecgeom::Vector3D<vecgeom::Precision> localpoint;
    vecgeom::Vector3D<vecgeom::Precision> localdir;
    // Impl::DoGlobalToLocalTransformation(in_state, globalpoint, globaldir, localpoint, localdir);
    vecgeom::Transformation3D m;
    in_state.TopMatrix(m);
    localpoint = m.Transform(globalpoint);
    localdir   = m.TransformDirection(globaldir);
    // The user may want to move point from boundary before computing the step
    localpoint += push * localdir;

    VPlacedVolumePtr_t hitcandidate = nullptr;
    vecgeom::Precision step = ComputeStepAndHit(localpoint, localdir, step_limit, in_state, out_state, hitcandidate);
    if (step < step_limit) step += push;

    if (out_state.IsOnBoundary()) {
      // Relocate the point after the step to refine out_state.
      localpoint += (step + kPush) * localdir;

      if (!hitcandidate) {
        // We didn't hit a daughter but instead we're exiting the current volume.
        RelocatePoint(localpoint, out_state);
      } else {
        // Otherwise check if we're directly entering other daughters transitively.
        localpoint = hitcandidate->GetTransformation()->Transform(localpoint);
        LocatePointIn(hitcandidate, localpoint, out_state, false);
      }

      if (out_state.Top() != nullptr) {
        while (out_state.Top()->IsAssembly() || out_state.GetNavIndex() == in_state.GetNavIndex()) {
          out_state.Pop();
        }
        assert(!out_state.Top()->GetLogicalVolume()->GetUnplacedVolume()->IsAssembly());
      }
    }

    return step;
  }

  // Computes a step from the globalpoint (which must be in the current volume)
  // into globaldir, taking step_limit into account. If a volume is hit, the
  // function calls out_state.SetBoundaryState(true) and
  //  - removes all volumes from out_state if the current volume is left, or
  //  - adds the hit daughter volume to out_state if one is hit.
  // However the function does _NOT_ relocate the state to the next volume,
  // that is entering multiple volumes that share a boundary.
  __host__ __device__ static double ComputeStepAndNextVolume(vecgeom::Vector3D<vecgeom::Precision> const &globalpoint,
                                                             vecgeom::Vector3D<vecgeom::Precision> const &globaldir,
                                                             vecgeom::Precision step_limit,
                                                             vecgeom::NavStateIndex const &in_state,
                                                             vecgeom::NavStateIndex &out_state,
                                                             vecgeom::Precision push = 0.)
  {
    // calculate local point/dir from global point/dir
    vecgeom::Vector3D<vecgeom::Precision> localpoint;
    vecgeom::Vector3D<vecgeom::Precision> localdir;
    // Impl::DoGlobalToLocalTransformation(in_state, globalpoint, globaldir, localpoint, localdir);
    vecgeom::Transformation3D m;
    in_state.TopMatrix(m);
    localpoint = m.Transform(globalpoint);
    localdir   = m.TransformDirection(globaldir);
    // The user may want to move point from boundary before computing the step
    localpoint += push * localdir;

    VPlacedVolumePtr_t hitcandidate = nullptr;
    vecgeom::Precision step = ComputeStepAndHit(localpoint, localdir, step_limit, in_state, out_state, hitcandidate);
    if (step < step_limit) step += push;

    if (out_state.IsOnBoundary()) {
      if (!hitcandidate) {
        vecgeom::VPlacedVolume const *currentmother       = out_state.Top();
        vecgeom::Vector3D<vecgeom::Precision> transformed = localpoint;
        // Push the point inside the next volume.
        static constexpr double kPush = 10. * vecgeom::kTolerance;
        transformed += (step + kPush) * localdir;
        do {
          out_state.SetLastExited();
          out_state.Pop();
          transformed   = currentmother->GetTransformation()->InverseTransform(transformed);
          currentmother = out_state.Top();
        } while (currentmother && (currentmother->IsAssembly() || !currentmother->UnplacedContains(transformed)));
      } else {
        out_state.Push(hitcandidate);
      }
    }

    return step;
  }

  // Relocate a state that was returned from ComputeStepAndNextVolume: It
  // recursively locates the pushed point in the containing volume.
  __host__ __device__ static void RelocateToNextVolume(vecgeom::Vector3D<vecgeom::Precision> &globalpoint,
                                                       vecgeom::Vector3D<vecgeom::Precision> const &globaldir,
                                                       vecgeom::NavStateIndex &state)
  {
    // Push the point inside the next volume.
    static constexpr double kPush = 10. * vecgeom::kTolerance;
    globalpoint += kPush * globaldir;

    // Calculate local point from global point.
    vecgeom::Transformation3D m;
    state.TopMatrix(m);
    vecgeom::Vector3D<vecgeom::Precision> localpoint = m.Transform(globalpoint);

    VPlacedVolumePtr_t pvol = state.Top();

    state.Pop();
    LocatePointIn(pvol, localpoint, state, false);

    if (state.Top() != nullptr) {
      while (state.Top()->IsAssembly()) {
        state.Pop();
      }
      assert(!state.Top()->GetLogicalVolume()->GetUnplacedVolume()->IsAssembly());
    }
  }
};

} // End namespace COPCORE_IMPL
#endif // RT_LOOP_NAVIGATOR_H_
