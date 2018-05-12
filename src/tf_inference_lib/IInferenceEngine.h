

#ifndef __IINFERENCEENGINE_H__
#define __IINFERENCEENGINE_H__

#include <cstdint>
#include <string>
#include <vector>

#define TF_INFERENCE_LIB_MAX_DIMS 10
namespace tf_interface_lib {

typedef enum {
	DT_FLOAT, DT_DOUBLE, DT_INT32,
	DT_UINT16,DT_UINT8, DT_INT16,
	DT_INT8, DT_STRING, DT_COMPLEX64,
	DT_COMPLEX128, DT_INT64, DT_BOOL,
	DT_QINT8, DT_QUINT8, DT_QINT16,
	DT_QUINT16, DT_QINT32, DT_BFLOAT16
} eExchangeDataType;


typedef struct {
	void* mem;
	int32_t data_len;
	int32_t dims[TF_INFERENCE_LIB_MAX_DIMS];
	eExchangeDataType data_type;
} tensor_exchange_t;



class IInferenceEngine {
public:
	IInferenceEngine(){};
	virtual ~IInferenceEngine(){};

	virtual int8_t init(std::string pathToModel) = 0;

	virtual int8_t infer() = 0;

	virtual void shutdown() = 0;

	virtual void* getOutputData(unsigned int index) = 0;
	virtual void* getInputData(unsigned int index) = 0;

	virtual void* addInput(const std::string blobName,
			const std::vector<int64_t>& dims,
			const eExchangeDataType dtype) = 0;

	virtual void addOutput(const std::string blobName,
			const std::vector<int64_t>& dims,
			const eExchangeDataType dtype) = 0;
};

class cInferenceEngineFactory {
public:
	cInferenceEngineFactory(){};
	virtual ~cInferenceEngineFactory();
	static IInferenceEngine* getInferenceEngine();
};

}

#endif // __IINFERENCEENGINE_H__
