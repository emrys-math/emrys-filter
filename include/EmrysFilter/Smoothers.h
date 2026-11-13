#pragma once

/** \file
 * Everything in the Smoothers layer: the histories of the Kalman, interacting multiple model,
 * %Gaussian mixture, particle, and hidden Markov model filters' forward passes, and the backward
 * passes over them.
 */

/** \defgroup Smoothers Smoothers
 *
 * The last layer (decisions.md [layered-headers]): estimates of past states from later data [smoothing]. A filter is
 * unchanged by smoothing but for predictJoint(), which fills the joint of a step's start and end, and the record of
 * the forward pass is the caller's: a history, which the caller keeps outside the filter and pushes each step to, and
 * whose smooth() takes the filter's latest state and fills every step's smoothed estimate, from the newest step back,
 * over the whole run or the newest steps of a Lag. GaussianHistory.h, with smoothStep(), the Rauch-Tung-Striebel step,
 * is the shape the others follow: HybridGaussianHistory.h smooths an ImmFilter by its modes, GaussianMixtureHistory.h a
 * MixtureFilter of Gaussians by its components' lineage, ParticleHistory.h a ParticleFilter by its particles'
 * ancestors, and CategoricalHistory.h a HiddenMarkovFilter, with its most probable sequence.
 */

#include <EmrysFilter/Smoothers/CategoricalHistory.h>
#include <EmrysFilter/Smoothers/GaussianHistory.h>
#include <EmrysFilter/Smoothers/GaussianMixtureHistory.h>
#include <EmrysFilter/Smoothers/HybridGaussianHistory.h>
#include <EmrysFilter/Smoothers/Lag.h>
#include <EmrysFilter/Smoothers/ParticleHistory.h>
