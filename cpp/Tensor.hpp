#ifndef __TENSOR_HPP__
#define __TENSOR_HPP__

#include <algorithm>
#include <cstring>
#include <iostream>
#include <list>
#include <vector>
#include <iterator>

#include "CompGraph/Abs.hpp"
#include "CompGraph/Add.hpp"
#include "CompGraph/CompGraph.hpp"
#include "CompGraph/Div.hpp"
#include "CompGraph/Exp.hpp"
#include "CompGraph/Log.hpp"
#include "CompGraph/Max.hpp"
#include "CompGraph/Mul.hpp"
#include "CompGraph/Number.hpp"
#include "CompGraph/Operation.hpp"
#include "CompGraph/Pow.hpp"
#include "CompGraph/Sigm.hpp"
#include "CompGraph/SmartPointer.hpp"
#include "CompGraph/Sub.hpp"

using namespace std;

template<typename T>
class Tensor
{
  public:
    Tensor()
      : shape_(0)
      , size_(0)
      , data_(nullptr)
      , owner_(true)
    {
    }

    explicit Tensor(const size_t& size)
      : shape_({ size })
      , size_(size)
      , owner_(true)
    {
        data_ = new SmartPointer<Operation<T>>[size];
        for (size_t i = 0; i < size_; i++)
            data_[i] = new Number<T>();
    }

    Tensor(const vector<size_t>& shape, bool init = true)
      : shape_(shape)
      , owner_(true)
    {
        size_t size = shape.size() == 0 ? 0 : 1;
        for (auto dim : shape) {
            size *= dim;
        }
        data_ = new SmartPointer<Operation<T>>[size];
        size_ = size;
        if (init) {
            for (size_t i = 0; i < size_; i++)
                data_[i] = new Number<T>();
        }
    }

    Tensor(const vector<T>& data)
      : shape_({ data.size() })
      , size_(data.size())
      , owner_(true)
    {
        data_ = new SmartPointer<Operation<T>>[size_];
        for (size_t i = 0; i < size_; i++)
            data_[i] = new Number<T>(data[i]);
    }

    void operator=(const Tensor<T>& other)
    {
        if (&other == this)
            return;
        if (owner_) {
            if (data_) {
                delete[] data_;
            }

            data_ = new SmartPointer<Operation<T>>[other.size_];
            other.copy_data(*this);
        } else {
#ifdef DEBUG
            if (other.size_ != size_) {
                throw length_error("Cannot modify shape of subarray.");
            }
#endif
            other.copy_data(*this);
        }

        this->shape_ = other.shape_;
        this->size_ = other.size_;
    }

    Tensor(const Tensor<T>& other)
    {
        if (&other == this)
            return;
        data_ = new SmartPointer<Operation<T>>[other.size_];
        this->size_ = other.size_;
        this->shape_ = other.shape_;
        this->owner_ = true;
        other.copy_data(*this);
    }

    ~Tensor()
    {
        if (owner_ && data_) {
            delete[] data_;
        }
    }

    // Iteration
    SmartPointer<Operation<double>>* begin() { return data_; }
    SmartPointer<Operation<double>>* end() { return data_ + size_; }

    // operator int() = delete;
    // operator int() const = delete;

    inline size_t size() const { return shape_.size() == 0 ? 0 : shape_[0]; }

    inline const vector<size_t>& shape() const { return shape_; }

    bool operator==(const vector<T>& other)
    {
        if (this->shape_.size() !=1 ) return false;
        for (size_t i = 0; i < this->size_; i++) {
            if (other[i] != this->data_[i]->value) return false;
        }
        return true;
    }

    Tensor<T> operator[](const size_t& index) const
    {
#ifdef DEBUG
        if (index >= this->shape_[0]) {
            throw out_of_range("Index out of range.");
        }
#endif
        size_t new_size = this->size_ / this->shape_[0];
        return Tensor(vector<size_t>(this->shape_.begin() + 1, this->shape_.end()),
                      new_size,
                      this->data_ + (index * new_size));
    }

    Tensor<T> operator[](const vector<size_t>& indices) const
    {
        size_t new_size = this->size_;
        size_t offset = 0;
        for (size_t i = 0; i < indices.size(); i++) {
            new_size = new_size / this->shape_[i];
            offset += indices[i] * new_size;
        }
        return Tensor(vector<size_t>(this->shape_.begin() + indices.size(), this->shape_.end()),
                      new_size,
                      this->data_ + offset);
    }

    // Access operator to get an element directly
    T& operator()(const size_t& index) const
    {
#ifdef DEBUG
        // Check if the index is valid
        if (shape_.size() > 1) {
            throw out_of_range("Trying to index multi-dimensional array");
        }
        if (index >= size_) {
            throw out_of_range("Index out of range");
        }
#endif

        return data_[index]->value;
    }

    T& operator()(const vector<size_t>& indices) const
    {
        size_t index = idx_from_v(indices);
        return data_[index]->value;
    }

    // operator T() const { return this->data_[0]->value; }
    operator T&() { return this->data_[0]->value; }

    void operator=(vector<T> v)
    {
#ifdef DEBUG
        if (this->shape_.size() > 1) {
            throw out_of_range("Cannot affect singular values to multi-dimensional tensor.");
        }
#endif
        if(owner_ && v.size() != this->shape_[0]) {
            shape_[0] = v.size();
            size_ = shape_[0];
            delete[] data_;
            data_ = new SmartPointer<Operation<T>>[v.size()];
        }

        for (size_t i = 0; i < shape_[0]; i++) {
            data_[i] = new Number(v[i]);
        }
    }

    void print() const
    {
        cout << "[";
        for (size_t i = 0; i < this->size_; ++i) {
            cout << this->data_[i]->value;
            if (i != this->size_ - 1) {
                cout << ", ";
            }
        }
        cout << "]\n";
    }

    void print_shape() const
    {
        cout << "Shape ([";
        for (size_t i = 0; i < shape_.size(); ++i) {
            cout << shape_[i];
            if (i != shape_.size() - 1) {
                cout << ", ";
            }
        }
        cout << "])\n";
    }

    inline bool empty() const { return this->size_ == 0; }

    inline size_t total_size() const { return this->size_; }

    void copy_data(Tensor& other) const
    {
        for (size_t i = 0; i < this->size_; i++) {
            other.data_[i] = data_[i];
        }
    }

    Tensor<T> sum() const
    {
        SmartPointer<Operation<T>> x = new Number<T>();
        for (size_t i = 0; i < size_; i++) {
            x = new Add<T>(x, data_[i]);
            x->forward();
        }
        Tensor<T> result(vector<size_t>{ 1 }, false);
        result.data_[0] = x;
        return result;
    }

    void add_dimension()
    {
        vector<size_t> new_shape = vector<size_t>(shape_.size() + 1);
        new_shape[0] = 1;
        copy(shape_.begin(), shape_.end(), new_shape.begin() + 1);
        shape_ = new_shape;
    }

    void reset_data()
    {
        for (size_t i = 0; i < size_; i++) {
            data_[i] = new Number<T>();
        }
    }

    Tensor<T> flatten()
    {
        Tensor<T> flat(this->size_);
        this->copy_data(flat);
        return flat;
    }

    void operator+=(const T& value)
    {
        SmartPointer<Operation<T>> val = new Number<T>(value);
        for (size_t i = 0; i < size_; i++) {
            data_[i] = new Add<T>(data_[i], val);
            data_[i]->forward();
        }
    }

    void operator+=(const Tensor<T>& other)
    {
#ifdef DEBUG
        if (other.size_ != this->size_) {
            fprintf(
              stderr,
              "Tensor should have the size in addition. Broadcasting is not supported yet.\n");
            exit(0);
        }
#endif
        for (size_t i = 0; i < size_; i++) {
            this->data_[i] = new Add(this->data_[i], other.data_[i]);
            this->data_[i]->forward();
        }
    }

    void operator-=(const T& value)
    {
        SmartPointer<Operation<T>> val = new Number<T>(value);
        for (size_t i = 0; i < size_; i++) {
            data_[i] = new Sub<T>(data_[i], val);
            data_[i]->forward();
        }
    }

    void operator-=(const Tensor<T>& other)
    {
#ifdef DEBUG
        if (other.size_ != this->size_) {
            fprintf(
              stderr,
              "Tensor should have the size in substraction. Broadcasting is not supported yet.\n");
            exit(0);
        }
#endif
        for (size_t i = 0; i < size_; i++) {
            this->data_[i] = new Sub<T>(this->data_[i], other.data_[i]);
            this->data_[i]->forward();
        }
    }

    void operator*=(const T& value)
    {
        SmartPointer<Operation<T>> val = new Number<T>(value);
        size_t nb = total_size();
        for (size_t i = 0; i < nb; i++) {
            data_[i] = new Mul<T>(data_[i], val);
            data_[i]->forward();
        }
    }

    void operator*=(const Tensor<T>& other)
    {
#ifdef DEBUG
        if (other.size_ != this->size_) {
            printf("Tensor should have the size in multiplication. Broadcasting is not supported "
                   "yet.\n");
            exit(0);
        }
#endif
        for (size_t i = 0; i < size_; i++) {
            this->data_[i] = new Mul<T>(this->data_[i], other.data_[i]);
            this->data_[i]->forward();
        }
    }

    void operator/=(const T& value)
    {
        SmartPointer<Operation<T>> val = new Number<T>(value);
        size_t nb = total_size();
        for (size_t i = 0; i < nb; i++) {
            data_[i] = new Div<T>(data_[i], val);
            data_[i]->forward();
        }
    }

    void operator/=(const Tensor<T>& other)
    {
#ifdef DEBUG
        if (other.size_ != this->size_ and other.total_size() != 1) {
            printf("Tensor should have the same size in division. Broadcasting is not supported "
                   "yet.\n");
            exit(0);
        }
#endif
        if (other.total_size() == 1) {
            for (size_t i = 0; i < size_; i++) {
                this->data_[i] = new Div<T>(this->data_[i], other.data_[0]);
                this->data_[i]->forward();
            }
        } else {
            for (size_t i = 0; i < size_; i++) {
                this->data_[i] = new Div<T>(this->data_[i], other.data_[i]);
                this->data_[i]->forward();
            }
        }
    }

    Tensor<T> operator+(const T& value) const
    {
        Tensor<T> result(*this);
        result += value;
        return result;
    }

    Tensor<T> operator+(const Tensor<T>& other) const
    {
        Tensor<T> result(shape_, false);
        for (size_t i = 0; i < result.size_; i++) {
            result.data_[i] = new Add<T>(data_[i], other.data_[i]);
            result.data_[i]->forward();
        }
        return move(result);
    }

    Tensor<T> operator-(const T& value) const
    {
        Tensor<T> result(*this);
        result -= value;
        return move(result);
    }

    Tensor<T> operator-(const Tensor<T>& other) const
    {
        Tensor<T> result(shape_);
        for (size_t i = 0; i < size_; i++) {
            result.data_[i] = new Sub<T>(data_[i], other.data_[i]);
            result.data_[i]->forward();
        }
        return move(result);
    }

    Tensor<T> operator-() const
    {
        Tensor<T> result(shape_);
        result -= *this;
        return move(result);
    }

    Tensor<T> operator*(const T& value) const
    {
        Tensor<T> result(*this);
        result *= value;
        return move(result);
    }

    Tensor<T> operator*(const Tensor<T>& other) const
    {
        Tensor<T> result(shape_);
        for (size_t i = 0; i < size_; i++) {
            result.data_[i] = new Mul<T>(data_[i], other.data_[i]);
            result.data_[i]->forward();
        }
        return move(result);
    }

    Tensor<T> operator/(const T& value) const
    {
        Tensor<T> result(*this);
        result /= value;
        return move(result);
    }

    Tensor<T> operator/(const Tensor<T>& other)
    {
        Tensor<T> result(shape_);
        for (size_t i = 0; i < size_; i++) {
            result.data_[i] = new Div<T>(data_[i], other.data_[i]);
            result.data_[i]->forward();
        }
        return move(result);
    }

    Tensor<T> pow(const T& value) const
    {
        Tensor<T> result(shape_, false);
        size_t nb = total_size();
        SmartPointer<Operation<T>> val = new Number<T>(value);
        for (size_t i = 0; i < nb; i++) {
            result.data_[i] = new Pow<T>(data_[i], val);
            result.data_[i]->forward();
        }
        return move(result);
    }

    Tensor<T> pow(const Tensor<T>& other) const
    {
#ifdef DEBUG
        if (other.size_ != this->size_) {
            printf("Tensor should have the size in power operation. Broadcasting is not supported "
                   "yet.\n");
            exit(0);
        }
#endif

        Tensor<T> result(shape_, false);
        size_t nb = total_size();
        for (size_t i = 0; i < nb; i++) {
            result.data_[i] = new Pow<T>(data_[i], other.data_[i]);
            result.data_[i]->forward();
        }
        return move(result);
    }

    Tensor<T> max(const T& value) const
    {
        Tensor<T> result(shape_, false);
        size_t nb = total_size();
        SmartPointer<Operation<T>> val = new Number<T>(value);
        for (size_t i = 0; i < nb; i++) {
            result.data_[i] = new Max<T>(data_[i], val);
            result.data_[i]->forward();
        }
        return move(result);
    }

    Tensor<T> max(const Tensor<T>& other) const
    {
#ifdef DEBUG
        if (other.size_ != this->size_) {
            printf("Tensor should have the size in max. Broadcasting is not supported "
                   "yet.\n");
            exit(0);
        }
#endif
        Tensor<T> result(shape_, false);
        size_t nb = total_size();
        for (size_t i = 0; i < nb; i++) {
            result.data_[i] = new Max<T>(data_[i], other.data_[i]);
            result.data_[i]->forward();
        }
        return move(result);
    }

    Tensor<T> log() const
    {
        Tensor<T> result(shape_, false);
        size_t nb = total_size();
        for (size_t i = 0; i < nb; i++) {
            result.data_[i] = new Log<T>(data_[i]);
            result.data_[i]->forward();
        }
        return move(result);
    }

    Tensor<T> exp() const
    {
        Tensor<T> result(shape_, false);
        size_t nb = total_size();
        for (size_t i = 0; i < nb; i++) {
            result.data_[i] = new Exp<T>(data_[i]);
            result.data_[i]->forward();
        }
        return move(result);
    }

    Tensor<T> abs() const
    {
        Tensor<T> result(shape_, false);
        size_t nb = total_size();
        for (size_t i = 0; i < nb; i++) {
            result.data_[i] = new Abs<T>(data_[i]);
            result.data_[i]->forward();
        }
        return move(result);
    }

    Tensor<T> sigm() const
    {
        Tensor<T> result(shape_, false);
        size_t nb = total_size();
        for (size_t i = 0; i < nb; i++) {
            result.data_[i] = new Sigm<T>(data_[i]);
            result.data_[i]->forward();
        }
        return move(result);
    }

    Tensor<T> transpose() const
    {
#ifdef DEBUG
        if (this->shape_.size() != 2) {
            fprintf(stderr, "Matrix transpose is only available for 2D tensors for now. Got :\n");
            this->print_shape();
            exit(0);
        }
#endif
        size_t h = this->shape_[0];
        size_t w = this->shape_[1];
        Tensor<T> result(vector<size_t>({ w, h }), false);
        for (size_t i = 0; i < h; i++) {
            for (size_t j = 0; j < w; j++) {
                result.data_[j * h + i] = this->data_[i * w + j];
            }
        }
        return move(result);
    }

    Tensor<T> mm(const Tensor<T>& other) const
    {
#ifdef DEBUG
        if (this->shape_.size() != 2 || other.shape_.size() != 2) {
            fprintf(stderr, "Matrix multiply is only available for 2D tensors for now. Got :\n");
            this->print_shape();
            other.print_shape();
            exit(0);
        }
#endif
        size_t h = this->shape_[0];
        size_t m = other.shape_[1];
#ifdef DEBUG
        size_t w = this->shape_[1];
        size_t n = other.shape_[0];
        if (w != n) {
            fprintf(
              stderr,
              "Error in matrix multiply : incompatible dimensions. ((%zu,%zu) and (%zu,%zu))\n",
              h,
              w,
              n,
              m);
            exit(0);
        }
#endif
        Tensor<T> result(
          vector<size_t>({ h, m }),
          false); // Initialize result with required dimensions, but don't populate with data.

        auto& lhs_data = this->data_;
        auto& rhs_data = other.data_;

        for (size_t i = 0; i < h; ++i) {
            for (size_t j = 0; j < m; ++j) {
                SmartPointer<Operation<T>> sum = new Number<T>();
                for (size_t k = 0; k < this->shape_[1]; ++k) {
                    SmartPointer<Operation<T>> product =
                      new Mul<T>(lhs_data[i * this->shape_[1] + k], rhs_data[k * m + j]);
                    product->forward();
                    sum = new Add<T>(sum, product);
                    sum->forward();
                }
                result.data_[i * m + j] = sum;
            }
        }
        return result;
    }

    Tensor<T> bmm(const Tensor<T>& other) const {
        size_t b = other.shape_[0];
        #ifdef DEBUG
        if(b != this->shape_[0]) {
            fprintf(stderr, "Error : unequal batch sizes in batched matrix multiply : got %zu and %zu.", this->shape_[0], b);
            exit(0);
        }
        #endif
        vector<size_t> shape = {this->shape_[1], other.shape_[2]};
        shape.insert(shape.begin(), b);
        Tensor<T> result(shape);
        for(size_t i = 0; i < b; i++) {
            result[i] = (*this)[i].mm(other[i]);
        }

        return result;
    }

    CompGraph<T> get_graph()
    {
#ifdef DEBUG
        if (size_ != 1) {
            fprintf(stderr, "Graph can only be computed if tensor has size 1.\n");
            exit(0);
        }
#endif
        return CompGraph<T>(this->data_[0]);
    }

    // void accumulate_gradients()
    // {
    //     for (size_t i = 0; i < size_; i++) {
    //         data_[i]->acc += data_[i]->gradient;
    //         data_[i]->gradient = 0;
    //     }
    // }

    T gradient(size_t index) { return data_[index]->gradient; }
    // T acc(size_t index) { return data_[index]->acc; }

    T gradient(vector<size_t> indices) { return data_[idx_from_v(indices)]->gradient; }
    // T acc(vector<size_t> indices) { return data_[idx_from_v(indices)]->acc; }

    void reset_gradients()
    {
        for (size_t i = 0; i < size_; i++) {
            // data_[i]->acc = 0;
            data_[i]->gradient = 0;
        }
    }

  private:
    vector<size_t> shape_;
    size_t size_;
    SmartPointer<Operation<T>>* data_;
    bool owner_;

    Tensor(const vector<size_t>& shape, const size_t& size, SmartPointer<Operation<T>>* const data)
      : shape_(shape)
      , size_(size)
      , data_(data)
      , owner_(false)
    {
    }

    size_t idx_from_v(const vector<size_t>& indices) const
    {
        // Calculate the linear index from the multi-dimensional indices
        size_t index = 0;
        size_t stride = 1;
        // Trick to get i >= 0 condition on size_t :)
        for (size_t i = shape_.size(); i--;) {
            // printf("i = %zu, stride = %zu, idx = %zu\n", i, stride, index);
            index += indices[i] * stride;
            stride *= shape_[i];
        }

#ifdef DEBUG
        // Check if the index is valid
        if (index >= size_) {
            throw out_of_range("Index out of range");
        }
#endif
        return index;
    }
};

#endif // __TENSOR_HPP__
