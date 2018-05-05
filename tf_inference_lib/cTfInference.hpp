/*
 * cTfInference.h
 *
 *  Created on: 18.02.2017
 *      Author: jan
 */

#ifndef CTFINFERENCE_HPP_
#define CTFINFERENCE_HPP_


#include <fstream>
#include <vector>

#include "IInferenceEngine.h"

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tf_interface_lib {


class cTfInference : public IInferenceEngine{
public:
	cTfInference();
	~cTfInference();

	int8_t init(std::string pathToModel);

	int8_t infer(tensor_exchange_t inputs[], uint8_t num_inputs,
			tensor_exchange_t outputs[], uint8_t num_outputs);

	void shutdown(){};

	void addInput(const std::string blobName,
			const std::vector<int64_t>& dims,
			const eExchangeDataType dtype,
			tensor_exchange_t& o_exchangeStruct);

	void addOutput(const std::string blobName,
				const std::vector<int64_t>& dims,
				const eExchangeDataType dtype,
				tensor_exchange_t& o_exchangeStruct);


	static tensorflow::Status copyDataIntoTensor(tensorflow::Tensor& dst, const tensor_exchange_t& src);


protected: /* helper functions */
	tensorflow::Status LoadGraph(std::string graph_file_name,
	                 std::unique_ptr<tensorflow::Session>* session);

    tensorflow::DataType convertDataType(eExchangeDataType exchangeType);



    template <typename T> static tensorflow::Status copyDataIntoTensorInternal(tensorflow::Tensor& dst, const tensor_exchange_t& src);

private: /* private members */
	bool m_bInitialized;
	std::unique_ptr<tensorflow::Session> m_pSession;
	std::vector<std::pair<std::string, tensorflow::Tensor> > m_vInputs;
	std::vector<tensorflow::Tensor> m_vOutputTensors;
	std::vector<tensorflow::string> m_vOutputNames;
};

}

#endif /* CTFINFERENCE_HPP_ */
