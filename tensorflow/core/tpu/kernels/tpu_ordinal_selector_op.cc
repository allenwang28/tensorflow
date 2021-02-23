#include <memory>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"


namespace tensorflow {
namespace {

// TPUOrdinalSelectorOp is a no-op for backward compatibility. The core
// selection algorithm happens inside TPUPartitionedCall.
class TPUOrdinalSelectorOp : public OpKernel {
 public:
  explicit TPUOrdinalSelectorOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  ~TPUOrdinalSelectorOp() override {}

  void Compute(OpKernelContext* ctx) override {
    Tensor output(DT_INT32, TensorShape({}));
    output.flat<int>().setValues({tpu::kDeferredCoreSelectionReserved});
    ctx->set_output(0, output);
    ctx->SetStatus(Status::OK());
  }

  bool IsExpensive() override { return false; }
};

}  // namespace

REGISTER_KERNEL_BUILDER(Name("TPUOrdinalSelector").Device(DEVICE_CPU),
                        TPUOrdinalSelectorOp);

REGISTER_KERNEL_BUILDER(Name("TPURoundRobin").Device(DEVICE_CPU),
                        TPUOrdinalSelectorOp);

}  // namespace tensorflow
