"""IR-level tests for 6-arg `tir.tvm_access_ptr` construction and roundtrip behavior.

These tests validate:
- 6-arg call construction and default sentinel pipeline (-1)
- Explicit pipeline encoding
- Valid pipeline value coverage and invalid-value rejection at API boundary
- rw_mask bitmask placement
- Buffer.access_ptr() roundtrip through PrimFunc + TIR passes
"""

import importlib

pytest = importlib.import_module("pytest")
tvm = importlib.import_module("tvm")
tir = tvm.tir


def _make_ptr_inputs():
    ptype = tvm.runtime.DataType("float16")
    data = tir.Var("buf", dtype="handle")
    return ptype, data


def _collect_access_ptr_calls(func: tir.PrimFunc):
    calls = []

    def _visit(node):
        if isinstance(node, tir.Call) and isinstance(node.op, tvm.ir.Op):
            if node.op.name == "tir.tvm_access_ptr":
                calls.append(node)

    tvm.tir.stmt_functor.post_order_visit(func.body, _visit)
    return calls


def test_access_ptr_6arg_default_sentinel():
    ptype, data = _make_ptr_inputs()
    access_call = tir.tvm_access_ptr(ptype, data, 0, 1024, 1)

    assert isinstance(access_call, tir.Call)
    assert len(access_call.args) == 6
    assert isinstance(access_call.args[5], tir.IntImm)
    assert access_call.args[5].value == -1


def test_access_ptr_6arg_explicit_pipeline():
    ptype, data = _make_ptr_inputs()
    access_call = tir.tvm_access_ptr(ptype, data, 0, 1024, 1, pipeline=3)

    assert isinstance(access_call, tir.Call)
    assert len(access_call.args) == 6
    assert isinstance(access_call.args[5], tir.IntImm)
    assert access_call.args[5].value == 3


def test_access_ptr_6arg_all_pipeline_values():
    ptype, data = _make_ptr_inputs()

    for val in [-1, 0, 1, 2, 3, 4, 5, 6, 7]:
        access_call = tir.tvm_access_ptr(ptype, data, 0, 1024, 1, pipeline=val)
        assert len(access_call.args) == 6
        assert isinstance(access_call.args[5], tir.IntImm)
        assert access_call.args[5].value == val

    for invalid in [object()]:
        with pytest.raises((TypeError, tvm.TVMError)):
            tir.tvm_access_ptr(ptype, data, 0, 1024, 1, pipeline=invalid)


def test_access_ptr_rw_mask_bitmask():
    ptype, data = _make_ptr_inputs()

    call1 = tir.tvm_access_ptr(ptype, data, 0, 1024, 1)
    assert call1.args[4].value == 1

    call2 = tir.tvm_access_ptr(ptype, data, 0, 1024, 2)
    assert call2.args[4].value == 2

    call3 = tir.tvm_access_ptr(ptype, data, 0, 1024, 3)
    assert call3.args[4].value == 3


def test_access_ptr_6arg_in_primfunc():
    @tir.prim_func
    def main(a: tir.handle):
        buf = tir.match_buffer(a, (1024,), dtype="float16", name="test_buf")
        tir.evaluate(buf.access_ptr("r"))

    mod = tvm.IRModule({"main": main})
    tir_text = mod.script()
    assert "tvm_access_ptr" in tir_text

    original_calls = _collect_access_ptr_calls(main)
    assert len(original_calls) == 1
    assert len(original_calls[0].args) == 6
    assert original_calls[0].args[5].value == -1

    transformed = tir.transform.Simplify()(mod)["main"]
    transformed_calls = _collect_access_ptr_calls(transformed)
    assert len(transformed_calls) == 1
    assert len(transformed_calls[0].args) == 6
    assert transformed_calls[0].args[5].value == -1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
