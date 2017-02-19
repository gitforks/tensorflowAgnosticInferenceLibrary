/*
 * cInferenceEngineFactory.cpp
 *
 *  Created on: 18.02.2017
 *      Author: jan
 */


#include "IInferenceEngine.h"
#include "cTfInference.hpp"

using namespace tf_interface_lib;

IInferenceEngine* cInferenceEngineFactory::getInferenceEngine()
{
	return new cTfInference();
	return nullptr;
}
