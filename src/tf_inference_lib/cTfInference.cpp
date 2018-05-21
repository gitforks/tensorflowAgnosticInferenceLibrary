/*
 * cTfInference.cpp
 *
 *  Created on: 18.02.2017
 *      Author: jan
 */

#include "cTfInference.hpp"
#include "tensorflow/c/c_api.h"

using namespace tf_interface_lib;
using tensorflow::string;
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::int32;

cTfInference::cTfInference()
{
        TF_Status* statusLibraryLoader = TF_NewStatus();
	TF_LoadLibrary("_trt_engine_op.so", statusLibraryLoader);
        if(TF_GetCode(statusLibraryLoader) != TF_OK)
        {
            std::cout << "TensorRT: " << TF_Message(statusLibraryLoader) << std::endl;
        }
        else
        {
            std::cout << "TensorRT successfully loaded!" << std::endl;
        }
	m_bInitialized = false;
}

cTfInference::~cTfInference()
{
}


tensorflow::int8 cTfInference::init(std::string pathToModel, float memory_fraction, bool allow_memory_growth)
{
	LOG(INFO) << "Loading Modell from: " << pathToModel;
	// We need to call this to set up global state for TensorFlow.
	// create pseudo call

	tensorflow::port::InitMain("tfInferenceLib", nullptr, nullptr);
        
	//sanitize memory_fraction
	if (memory_fraction < 0.0 || memory_fraction > 1.0)
	{
		LOG(ERROR) << "Memory fraction has to be between 0.0 and 1.0! Inference engine will not be initialized." << std::endl;
		return -1;
	}
	// First we load and initialize the model.
	string graph_path = tensorflow::io::JoinPath(pathToModel);
	Status load_graph_status = LoadGraph(graph_path, &m_pSession, memory_fraction, allow_memory_growth);
	if (!load_graph_status.ok()) {
		LOG(ERROR) << load_graph_status;
		return -1;
	}
	m_bInitialized = true;
	return 0;
}

void* cTfInference::addInput(const std::string blobName,
		const std::vector<int64_t>& dims,
		const eExchangeDataType dtype)
{
	std::vector<tensorflow::int64> tf_dims(dims.begin(), dims.end());

	Tensor inp(convertDataType(dtype), tensorflow::TensorShape(tf_dims));
	m_vInputs.push_back(
		    		{blobName,
		    	     inp
		    		});

	LOG(INFO) << "Adding input \"" << blobName << "\"with properties " << inp.DebugString();

	return getDataPointer(inp);


}
void cTfInference::addOutput(const std::string blobName,
		const std::vector<int64_t>& dims,
		const eExchangeDataType dtype)
{
	std::vector<tensorflow::int64> tf_dims(dims.begin(), dims.end());

	Tensor outp(convertDataType(dtype), tensorflow::TensorShape(tf_dims));
	m_vOutputTensors.push_back(outp);
	m_vOutputNames.push_back(blobName); // TODO: make safe by avoiding reading more than 255 chars

	LOG(INFO) << "Adding output \"" << blobName << "\" with properties " << outp.DebugString();

}

void* cTfInference::getInputData(unsigned int index)
{
	if (index < m_vInputs.size())
	{
		//return m_vInputs.at(index).second.flat<float>().data();
		return getDataPointer(m_vInputs.at(index).second);
	}
	else
	{
		return nullptr;
	}
}

void* cTfInference::getOutputData(unsigned int index)
{
	if (index < m_vOutputTensors.size())
	{
		return getDataPointer(m_vOutputTensors.at(index));
	}
	else
	{
		return nullptr;
	}
}


tensorflow::int8 cTfInference::infer()
{

	// Check if inference engine is initialized
	if(!m_bInitialized)
	{
		LOG(ERROR) << "Instance of cTfInference not initialized";
		return -1;
	}

	// infer through the graph
	Status run_status = m_pSession->Run(m_vInputs,
			m_vOutputNames, {}, &m_vOutputTensors);


	if (!run_status.ok()) {
		LOG(ERROR) << "Running model failed: " << run_status;
		return -1;
	}

	return 0;


}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status cTfInference::LoadGraph(string graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session,
		 float per_process_gpu_memory_fraction = 1.0,
                 bool allow_memory_growth = false) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  tensorflow::SessionOptions sessionOptions;

  sessionOptions.config.mutable_gpu_options()->set_allow_growth(allow_memory_growth);
  sessionOptions.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(per_process_gpu_memory_fraction);
  session->reset(tensorflow::NewSession(sessionOptions));
  Status session_create_status = (*session)->Create(graph_def);

  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

tensorflow::DataType cTfInference::convertDataType(eExchangeDataType exchangeType)
{
	switch(exchangeType)
	{
	case eExchangeDataType::DT_FLOAT: return tensorflow::DT_FLOAT; break;
	case eExchangeDataType::DT_DOUBLE: return tensorflow::DT_DOUBLE; break;
	case eExchangeDataType::DT_INT32: return tensorflow::DT_INT32; break;
	case eExchangeDataType::DT_UINT16: return tensorflow::DT_UINT16; break;
	case eExchangeDataType::DT_UINT8: return tensorflow::DT_UINT8; break;
	case eExchangeDataType::DT_INT16: return tensorflow::DT_INT16; break;
	case eExchangeDataType::DT_INT8: return tensorflow::DT_INT8; break;
	case eExchangeDataType::DT_STRING: return tensorflow::DT_STRING; break;
	case eExchangeDataType::DT_COMPLEX64: return tensorflow::DT_COMPLEX64; break;
	case eExchangeDataType::DT_COMPLEX128: return tensorflow::DT_COMPLEX128; break;
	case eExchangeDataType::DT_INT64: return tensorflow::DT_INT64; break;
	case eExchangeDataType::DT_BOOL: return tensorflow::DT_BOOL; break;
	case eExchangeDataType::DT_QINT8: return tensorflow::DT_QINT8; break;
	case eExchangeDataType::DT_QUINT8: return tensorflow::DT_QUINT8; break;
	case eExchangeDataType::DT_QINT16: return tensorflow::DT_QINT16; break;
	case eExchangeDataType::DT_QUINT16: return tensorflow::DT_QUINT16; break;
	case eExchangeDataType::DT_QINT32: return tensorflow::DT_QINT32; break;
	case eExchangeDataType::DT_BFLOAT16: return tensorflow::DT_BFLOAT16; break;
	}

	return tensorflow::DT_FLOAT;
}

void*  cTfInference::getDataPointer(tensorflow::Tensor& tensor)
{
	switch(tensor.dtype())
	{
	case tensorflow::DT_FLOAT: return tensor.flat<float>().data(); break;
	case tensorflow::DT_DOUBLE: return tensor.flat<double>().data(); break;
	case tensorflow::DT_INT32: return tensor.flat<int32_t>().data(); break;
	case tensorflow::DT_UINT16: return tensor.flat<uint16_t>().data(); break;
	case tensorflow::DT_UINT8: return tensor.flat<uint8_t>().data(); break;
	case tensorflow::DT_INT16: return tensor.flat<int16_t>().data(); break;
	case tensorflow::DT_INT8: return tensor.flat<int8_t>().data(); break;
	/*case tensorflow::DT_STRING: return tensorflow::DT_STRING; break;
	case tensorflow::DT_COMPLEX64: return tensorflow::DT_COMPLEX64; break;
	case tensorflow::DT_COMPLEX128: return tensorflow::DT_COMPLEX128; break;
	case tensorflow::DT_INT64: return tensorflow::DT_INT64; break;
	case tensorflow::DT_BOOL: return tensorflow::DT_BOOL; break;
	case tensorflow::DT_QINT8: return tensorflow::DT_QINT8; break;
	case tensorflow::DT_QUINT8: return tensorflow::DT_QUINT8; break;
	case tensorflow::DT_QINT16: return tensorflow::DT_QINT16; break;
	case tensorflow::DT_QUINT16: return tensorflow::DT_QUINT16; break;
	case tensorflow::DT_QINT32: return tensorflow::DT_QINT32; break;
	case tensorflow::DT_BFLOAT16: return tensorflow::DT_BFLOAT16; break;*/
	default:
		LOG(ERROR) << "Unsupported data type yet in tf_inference_lib";
		return nullptr;
	}

	return nullptr;
}

