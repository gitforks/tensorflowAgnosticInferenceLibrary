/*
 * cTfInference.cpp
 *
 *  Created on: 18.02.2017
 *      Author: jan
 */

#include "cTfInference.hpp"

using namespace tf_interface_lib;
using tensorflow::string;
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::int32;

cTfInference::cTfInference()
{
	// TODO Auto-generated constructor stub
	m_bInitialized = false;

}

cTfInference::~cTfInference()
{
	// TODO Auto-generated destructor stub
}


tensorflow::int8 cTfInference::init(std::string pathToModel)
{
	LOG(INFO) << "Loading Modell from: " << pathToModel;
	// We need to call this to set up global state for TensorFlow.
	// create pseudo call

	tensorflow::port::InitMain("tfInferenceLib", nullptr, nullptr);

	// First we load and initialize the model.
	string graph_path = tensorflow::io::JoinPath(pathToModel);
	Status load_graph_status = LoadGraph(graph_path, &m_pSession);
	if (!load_graph_status.ok()) {
		LOG(ERROR) << load_graph_status;
		return -1;
	}
	m_bInitialized = true;
	return 0;
}

void cTfInference::addInput(const std::string blobName,
		const std::vector<int64_t>& dims,
		const eExchangeDataType dtype,
		tensor_exchange_t& o_exchangeStruct)
{

	memset(&o_exchangeStruct, 0, sizeof(o_exchangeStruct));
	std::vector<tensorflow::int64> tf_dims(dims.begin(), dims.end());

	Tensor inp(convertDataType(dtype), tensorflow::TensorShape(tf_dims));
	m_vInputs.push_back(
		    		{blobName,
		    	     inp
		    		});

	LOG(INFO) << "Adding input \"" << blobName << "\"with properties " << inp.DebugString();

	o_exchangeStruct.data_type = dtype;
	for (unsigned int i=0; i<TF_INFERENCE_LIB_MAX_DIMS && i<dims.size(); i++)
	{
		o_exchangeStruct.dims[i] = dims.at(i);
	}
	o_exchangeStruct.data_len = 0;
	o_exchangeStruct.mem = nullptr;


}
void cTfInference::addOutput(const std::string blobName,
		const std::vector<int64_t>& dims,
		const eExchangeDataType dtype,
		tensor_exchange_t& o_exchangeStruct)
{
	memset(&o_exchangeStruct, 0, sizeof(o_exchangeStruct));
	std::vector<tensorflow::int64> tf_dims(dims.begin(), dims.end());

	Tensor outp(convertDataType(dtype), tensorflow::TensorShape(tf_dims));
	m_vOutputTensors.push_back(outp);
	m_vOutputNames.push_back(blobName); // TODO: make safe by avoiding reading more than 255 chars

	LOG(INFO) << "Adding output \"" << blobName << "\" with properties " << outp.DebugString();

	o_exchangeStruct.data_type = dtype;
	for (unsigned int i=0; i<TF_INFERENCE_LIB_MAX_DIMS && i<dims.size(); i++)
	{
		o_exchangeStruct.dims[i] = dims.at(i);
	}
	o_exchangeStruct.data_len = 0;
	o_exchangeStruct.mem = nullptr;
}

tensorflow::int8 cTfInference::infer(tensor_exchange_t inputs[], tensorflow::uint8 num_inputs,
		tensor_exchange_t outputs[], tensorflow::uint8 num_outputs)
{

	// Check if inference engine is initialized
	if(!m_bInitialized)
	{
		LOG(ERROR) << "Instance of cTfInference not initialized";
		return -1;
	}

	// Copy all inputs into a local tensor <tensorflow agnostic interface>
	for (int i= 0; i<num_inputs; i++)
	{
		copyDataIntoTensor(std::get<1>(m_vInputs.at(i)), inputs[i]);

	}

	// infer through the graph
	Status run_status = m_pSession->Run(m_vInputs,
			m_vOutputNames, {}, &m_vOutputTensors);

	// Provide the results in the tensorflow agnostic interface
	for (int i=0; i<num_outputs; i++)
	{
		if(!m_vOutputTensors[i].IsAligned()) // todo: how to enforce alignment?
		  {
			  LOG(ERROR) << "Output-Tensor is not aligned \n";
			  return -1;
		  }
		outputs[i].mem = (void*) m_vOutputTensors[i].tensor_data().data();
		outputs[i].data_len = m_vOutputTensors[i].tensor_data().size();
		// todo: update data type, too?
	}


	if (!run_status.ok()) {
		LOG(ERROR) << "Running model failed: " << run_status;
		return -1;
	}

	return 0;


}

Status cTfInference::copyDataIntoTensor(tensorflow::Tensor& dst, const tensor_exchange_t& src)
{
	switch(src.data_type)
		{
		case eExchangeDataType::DT_FLOAT: return copyDataIntoTensorInternal<float>(dst, src); break;
		case eExchangeDataType::DT_DOUBLE: return copyDataIntoTensorInternal<double>(dst, src); break;
		case eExchangeDataType::DT_INT32: return copyDataIntoTensorInternal<tensorflow::int32>(dst, src); break;
		case eExchangeDataType::DT_UINT16: return copyDataIntoTensorInternal<tensorflow::uint16>(dst, src); break;
		case eExchangeDataType::DT_UINT8: return copyDataIntoTensorInternal<tensorflow::uint8>(dst, src); break;
		case eExchangeDataType::DT_INT16: return copyDataIntoTensorInternal<tensorflow::int16>(dst, src); break;
		case eExchangeDataType::DT_INT8: return copyDataIntoTensorInternal<tensorflow::int8>(dst, src); break;
		case eExchangeDataType::DT_STRING: return tensorflow::errors::Unimplemented("String Tensors are not implemented"); break;
		case eExchangeDataType::DT_COMPLEX64: return copyDataIntoTensorInternal<tensorflow::complex64>(dst, src); break;
		case eExchangeDataType::DT_COMPLEX128: return copyDataIntoTensorInternal<tensorflow::complex128>(dst, src); break;
		case eExchangeDataType::DT_INT64: return copyDataIntoTensorInternal<tensorflow::int64>(dst, src); break;
		case eExchangeDataType::DT_BOOL: return copyDataIntoTensorInternal<tensorflow::uint8>(dst, src); break;
		case eExchangeDataType::DT_QINT8: return copyDataIntoTensorInternal<tensorflow::qint8>(dst, src); break;
		case eExchangeDataType::DT_QUINT8: return copyDataIntoTensorInternal<tensorflow::quint8>(dst, src); break;
		case eExchangeDataType::DT_QINT16: return copyDataIntoTensorInternal<tensorflow::qint16>(dst, src); break;
		case eExchangeDataType::DT_QUINT16: return copyDataIntoTensorInternal<tensorflow::quint16>(dst, src); break;
		case eExchangeDataType::DT_QINT32: return copyDataIntoTensorInternal<tensorflow::qint32>(dst, src); break;
		case eExchangeDataType::DT_BFLOAT16: return copyDataIntoTensorInternal<tensorflow::bfloat16>(dst, src); break;
		}

	return tensorflow::Status::OK();
}

template <typename T>
tensorflow::Status cTfInference::copyDataIntoTensorInternal(tensorflow::Tensor& dst, const tensor_exchange_t& src)
{
	if(src.mem != nullptr)
	{
		auto mappedTensor = dst.flat<T>(); // a flat array (Eigen)
		T* mappedMemory = (T*) (src.mem);
		void* memoryLimit = (char*) src.mem + src.data_len - sizeof(T);
		for (tensorflow::int64 j = 0;
				j<dst.NumElements() && (void*) mappedMemory <= memoryLimit;
				j++)
		{
			mappedTensor(j) = *mappedMemory;
			mappedMemory++;
		}
	}

	return tensorflow::Status::OK();
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status cTfInference::LoadGraph(string graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
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

