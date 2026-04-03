// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file rewrite_access_ptr.cc
 * \brief Rewrite tvm_access_ptr calls to tl_access_ptr calls.
 *
 * This pass runs early in OptimizeForTarget, replacing all
 * builtin::tvm_access_ptr() Call nodes with tl::tl_access_ptr()
 * Call nodes. Arguments are preserved verbatim.
 */

#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"

namespace tvm {
namespace tl {
using namespace tir;

class AccessPtrRewriter : public StmtExprMutator {
public:
  PrimExpr VisitExpr_(const CallNode* op) final {
    auto call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    auto* call_node = call.as<CallNode>();

    if (call_node->op.same_as(builtin::tvm_access_ptr())) {
      return Call(call_node->dtype, tl::tl_access_ptr(), call_node->args,
                  call_node->span);
    }
    return call;
  }
};

tir::transform::Pass RewriteAccessPtr() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* fptr = f.CopyOnWrite();
    AccessPtrRewriter rewriter;
    fptr->body = rewriter(std::move(fptr->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.RewriteAccessPtr", {});
}

TVM_REGISTER_GLOBAL("tl.transform.RewriteAccessPtr")
    .set_body_typed(RewriteAccessPtr);

}  // namespace tl
}  // namespace tvm
