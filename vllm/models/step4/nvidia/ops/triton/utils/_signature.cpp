#ifndef Py_LIMITED_API
#define Py_LIMITED_API 0x03100000  // Python 3.10+ limited API
#endif
#include <Python.h>

#include <chrono>
#include <cstdint>
#include <string>

namespace {

constexpr std::uint64_t kHashSeed = 1469598103934665603ull;
constexpr std::uint64_t kHashMagic = 0x9e3779b97f4a7c15ull;

std::uint64_t hash_combine(std::uint64_t seed, std::uint64_t value) {
    return seed ^ (value + kHashMagic + (seed << 6) + (seed >> 2));
}

std::uint64_t hash_string(std::uint64_t seed, const std::string& text) {
    for (unsigned char c : text) {
        seed = hash_combine(seed, static_cast<std::uint64_t>(c));
    }
    return seed;
}

std::uint64_t hash_tuple_ints(std::uint64_t seed, PyObject* tuple_obj) {
    Py_ssize_t n = PyTuple_Size(tuple_obj);
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* item = PyTuple_GetItem(tuple_obj, i);
        long long value = PyLong_AsLongLong(item);
        if (value == -1 && PyErr_Occurred()) {
            return 0;
        }
        seed = hash_combine(seed, static_cast<std::uint64_t>(value));
    }
    return seed;
}

PyObject* get_tensor_type() {
    static PyObject* tensor_type = nullptr;
    if (!tensor_type) {
        PyObject* torch_module = PyImport_ImportModule("torch");
        if (!torch_module) {
            return nullptr;
        }
        tensor_type = PyObject_GetAttrString(torch_module, "Tensor");
        Py_DECREF(torch_module);
        if (!tensor_type) {
            return nullptr;
        }
    }
    return tensor_type;
}

PyObject* get_size_type() {
    static PyObject* size_type = nullptr;
    if (!size_type) {
        PyObject* torch_module = PyImport_ImportModule("torch");
        if (!torch_module) {
            return nullptr;
        }
        size_type = PyObject_GetAttrString(torch_module, "Size");
        Py_DECREF(torch_module);
        if (!size_type) {
            return nullptr;
        }
    }
    return size_type;
}

PyObject* make_tagged_tuple(const char* tag, PyObject* values) {
    if (!values) {
        return nullptr;
    }
    PyObject* tag_str = PyUnicode_FromString(tag);
    if (!tag_str) {
        Py_DECREF(values);
        return nullptr;
    }
    PyObject* result = PyTuple_Pack(2, tag_str, values);
    Py_DECREF(tag_str);
    Py_DECREF(values);
    return result;
}

PyObject* convert_sequence(PyObject* seq);
PyObject* value_signature(PyObject* obj);
PyObject* dict_signature(PyObject* obj);

PyObject* dict_signature(PyObject* obj) {
    PyObject* items = PyMapping_Items(obj);
    if (!items) {
        return nullptr;
    }
    PyObject* item_list = PySequence_List(items);
    Py_DECREF(items);
    if (!item_list) {
        return nullptr;
    }
    if (PyList_Sort(item_list) < 0) {
        Py_DECREF(item_list);
        return nullptr;
    }
    Py_ssize_t n = PyList_Size(item_list);
    PyObject* converted = PyTuple_New(n);
    if (!converted) {
        Py_DECREF(item_list);
        return nullptr;
    }
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* pair = PyList_GetItem(item_list, i);
        if (!PyTuple_Check(pair) || PyTuple_Size(pair) != 2) {
            Py_DECREF(item_list);
            Py_DECREF(converted);
            PyErr_SetString(PyExc_TypeError, "dictionary item is not a 2-tuple");
            return nullptr;
        }
        PyObject* key = PyTuple_GetItem(pair, 0);
        PyObject* value = PyTuple_GetItem(pair, 1);
        PyObject* value_sig = value_signature(value);
        if (!value_sig) {
            Py_DECREF(item_list);
            Py_DECREF(converted);
            return nullptr;
        }
        PyObject* entry = PyTuple_Pack(2, key, value_sig);
        Py_DECREF(value_sig);
        if (!entry) {
            Py_DECREF(item_list);
            Py_DECREF(converted);
            return nullptr;
        }
        if (PyTuple_SetItem(converted, i, entry) < 0) {
            Py_DECREF(entry);
            Py_DECREF(item_list);
            Py_DECREF(converted);
            return nullptr;
        }
    }
    Py_DECREF(item_list);
    return converted;
}

PyObject* convert_sequence(PyObject* seq) {
    PyObject* seq_tuple = PySequence_Tuple(seq);
    if (!seq_tuple) {
        return nullptr;
    }
    Py_ssize_t n = PyTuple_Size(seq_tuple);
    PyObject* result = PyTuple_New(n);
    if (!result) {
        Py_DECREF(seq_tuple);
        return nullptr;
    }
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* item = PyTuple_GetItem(seq_tuple, i);
        PyObject* converted = value_signature(item);
        if (!converted) {
            Py_DECREF(seq_tuple);
            Py_DECREF(result);
            return nullptr;
        }
        if (PyTuple_SetItem(result, i, converted) < 0) {
            Py_DECREF(converted);
            Py_DECREF(seq_tuple);
            Py_DECREF(result);
            return nullptr;
        }
    }
    Py_DECREF(seq_tuple);
    return result;
}

PyObject* tensor_signature(PyObject* obj) {
    PyObject* dtype_obj = PyObject_GetAttrString(obj, "dtype");
    if (!dtype_obj) {
        return nullptr;
    }
    PyObject* dtype_str_obj = PyObject_Str(dtype_obj);
    Py_DECREF(dtype_obj);
    if (!dtype_str_obj) {
        return nullptr;
    }
    Py_ssize_t dtype_size = 0;
    const char* dtype_cstr = PyUnicode_AsUTF8AndSize(dtype_str_obj, &dtype_size);
    if (!dtype_cstr) {
        Py_DECREF(dtype_str_obj);
        return nullptr;
    }
    std::string dtype(dtype_cstr, static_cast<size_t>(dtype_size));
    Py_DECREF(dtype_str_obj);

    PyObject* shape_obj = PyObject_GetAttrString(obj, "shape");
    if (!shape_obj) {
        return nullptr;
    }
    PyObject* shape_tuple = PySequence_Tuple(shape_obj);
    Py_DECREF(shape_obj);
    if (!shape_tuple) {
        return nullptr;
    }

    PyObject* stride_obj = PyObject_CallMethod(obj, "stride", nullptr);
    if (!stride_obj) {
        Py_DECREF(shape_tuple);
        return nullptr;
    }
    PyObject* stride_tuple = PySequence_Tuple(stride_obj);
    Py_DECREF(stride_obj);
    if (!stride_tuple) {
        Py_DECREF(shape_tuple);
        return nullptr;
    }

    std::uint64_t hash_value = hash_string(kHashSeed, dtype);
    hash_value = hash_tuple_ints(hash_value, shape_tuple);
    if (PyErr_Occurred()) {
        Py_DECREF(shape_tuple);
        Py_DECREF(stride_tuple);
        return nullptr;
    }
    hash_value = hash_tuple_ints(hash_value, stride_tuple);
    if (PyErr_Occurred()) {
        Py_DECREF(shape_tuple);
        Py_DECREF(stride_tuple);
        return nullptr;
    }

    Py_DECREF(shape_tuple);
    Py_DECREF(stride_tuple);

    PyObject* tag = PyUnicode_FromString("tensor_hash");
    if (!tag) {
        return nullptr;
    }
    PyObject* value = PyLong_FromUnsignedLongLong(hash_value);
    if (!value) {
        Py_DECREF(tag);
        return nullptr;
    }
    PyObject* result = PyTuple_Pack(2, tag, value);
    Py_DECREF(tag);
    Py_DECREF(value);
    return result;
}

PyObject* value_signature(PyObject* obj) {
    if (obj == Py_None) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    if (PyBool_Check(obj) || PyLong_Check(obj) || PyFloat_Check(obj) || PyUnicode_Check(obj)) {
        Py_INCREF(obj);
        return obj;
    }

    PyObject* tensor_type = get_tensor_type();
    if (PyErr_Occurred()) {
        return nullptr;
    }
    if (tensor_type) {
        int is_tensor = PyObject_IsInstance(obj, tensor_type);
        if (is_tensor < 0) {
            return nullptr;
        }
        if (is_tensor) {
            return tensor_signature(obj);
        }
    }

    PyObject* size_type = get_size_type();
    if (PyErr_Occurred()) {
        return nullptr;
    }
    if (size_type) {
        int is_size = PyObject_IsInstance(obj, size_type);
        if (is_size < 0) {
            return nullptr;
        }
        if (is_size) {
            PyObject* size_tuple = PySequence_Tuple(obj);
            if (!size_tuple) {
                return nullptr;
            }
            return make_tagged_tuple("size", size_tuple);
        }
    }

    if (PyTuple_Check(obj)) {
        PyObject* tup = convert_sequence(obj);
        return make_tagged_tuple("tuple", tup);
    }
    if (PyList_Check(obj)) {
        PyObject* lst = convert_sequence(obj);
        return make_tagged_tuple("list", lst);
    }
    if (PyDict_Check(obj)) {
        PyObject* dict_sig = dict_signature(obj);
        return make_tagged_tuple("dict", dict_sig);
    }

    PyObject* type_obj = PyObject_Type(obj);
    if (!type_obj) {
        return nullptr;
    }
    PyObject* type_name = PyObject_GetAttrString(type_obj, "__name__");
    Py_DECREF(type_obj);
    if (!type_name) {
        return nullptr;
    }
    PyObject* repr = PyObject_Repr(obj);
    if (!repr) {
        Py_DECREF(type_name);
        return nullptr;
    }
    PyObject* result = PyTuple_Pack(2, type_name, repr);
    Py_DECREF(type_name);
    Py_DECREF(repr);
    return result;
}

}  // namespace

static PyObject* build_signature(PyObject* /*module*/, PyObject* args) {
    PyObject* grid_meta;
    PyObject* meta_args;
    PyObject* runtime_args;
    if (!PyArg_ParseTuple(args, "OOO", &grid_meta, &meta_args, &runtime_args)) {
        return nullptr;
    }
    (void)grid_meta;

#ifdef SIGNATURE_TIMER
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start_total = Clock::now();
    Clock::time_point start_runtime = start_total;
    Clock::time_point end_runtime;
    Clock::time_point end_meta;
#endif

    PyObject* runtime_sig = convert_sequence(runtime_args);
    if (!runtime_sig) {
        return nullptr;
    }

#ifdef SIGNATURE_TIMER
    end_runtime = Clock::now();
#endif

    PyObject* meta_sig = convert_sequence(meta_args);
    if (!meta_sig) {
        Py_DECREF(runtime_sig);
        return nullptr;
    }

#ifdef SIGNATURE_TIMER
    end_meta = Clock::now();
#endif

    Py_ssize_t runtime_size = PyTuple_Size(runtime_sig);
    Py_ssize_t meta_size = PyTuple_Size(meta_sig);
    PyObject* combined = PyTuple_New(runtime_size + meta_size);
    if (!combined) {
        Py_DECREF(runtime_sig);
        Py_DECREF(meta_sig);
        return nullptr;
    }

    for (Py_ssize_t i = 0; i < runtime_size; ++i) {
        PyObject* item = PyTuple_GetItem(runtime_sig, i);
        Py_INCREF(item);
        if (PyTuple_SetItem(combined, i, item) < 0) {
            Py_DECREF(item);
            Py_DECREF(runtime_sig);
            Py_DECREF(meta_sig);
            Py_DECREF(combined);
            return nullptr;
        }
    }
    for (Py_ssize_t i = 0; i < meta_size; ++i) {
        PyObject* item = PyTuple_GetItem(meta_sig, i);
        Py_INCREF(item);
        if (PyTuple_SetItem(combined, runtime_size + i, item) < 0) {
            Py_DECREF(item);
            Py_DECREF(runtime_sig);
            Py_DECREF(meta_sig);
            Py_DECREF(combined);
            return nullptr;
        }
    }

#ifdef SIGNATURE_TIMER
    Clock::time_point end_total = Clock::now();
    auto runtime_us = std::chrono::duration_cast<std::chrono::microseconds>(end_runtime - start_runtime).count();
    auto meta_us = std::chrono::duration_cast<std::chrono::microseconds>(end_meta - end_runtime).count();
    auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total).count();
    PySys_WriteStdout("[signature] runtime %lld us meta %lld us total %lld us\n",
                      static_cast<long long>(runtime_us),
                      static_cast<long long>(meta_us),
                      static_cast<long long>(total_us));
#endif

    Py_DECREF(runtime_sig);
    Py_DECREF(meta_sig);
    return combined;
}

static PyMethodDef SignatureMethods[] = {
    {"build_signature", build_signature, METH_VARARGS, "Build runtime/meta signature tuple"},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef signature_module = {
    PyModuleDef_HEAD_INIT,
    "_signature",
    "Fast signature builder for Triton driver cache",
    -1,
    SignatureMethods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

PyMODINIT_FUNC PyInit__signature(void) {
    return PyModule_Create(&signature_module);
}
