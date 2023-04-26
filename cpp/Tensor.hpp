#ifndef __TENSOR_HPP__
#define __TENSOR_HPP__

#include <algorithm>
#include <cstring>
#include <iostream>
#include <list>
#include <vector>

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

    Tensor(const size_t& size)
      : shape_({ size })
      , size_(size)
      , owner_(true)
    {
        data_ = new T[size];
        memset(data_, 0, sizeof(T) * size);
    }

    Tensor(const vector<size_t>& shape)
      : shape_(shape)
      , owner_(true)
    {
        size_t size = shape.size() == 0 ? 0 : 1;
        for (auto dim : shape) {
            size *= dim;
        }
        data_ = new T[size];
        memset(data_, 0, sizeof(T) * size);
        size_ = size;
    }

    Tensor(const vector<T>& data)
      : shape_({ data.size() })
      , size_(data.size())
      , owner_(true)
    {
        data_ = new T[size_];
        memcpy(data_, data.data(), size_ * sizeof(T));
    }

    void operator=(const Tensor<T>& other)
    {
        if (&other == this)
            return;
        if (owner_) {
            delete[] data_;
            data_ = new T[other.size_];
            memcpy(this->data_, other.data_, other.size_ * sizeof(T));
        } else {
            if (other.size_ != size_) {
                throw length_error("Cannot modify shape of subarray.");
            }
            memcpy(this->data_, other.data_, other.size_ * sizeof(T));
        }
        this->shape_ = other.shape_;
        this->size_ = other.size_;
    }

    Tensor(const Tensor<T>& other)
    {
        if (&other == this)
            return;
        data_ = new T[other.size_];
        memcpy(this->data_, other.data_, other.size_ * sizeof(T));
        this->shape_ = other.shape_;
        this->size_ = other.size_;
        this->owner_ = true;
    }

    ~Tensor()
    {
        if (owner_ && data_)
            delete[] data_;
    }

    operator int() = delete;
    operator int() const = delete;

    size_t size() const { return shape_[0]; }

    const vector<size_t>& shape() const { return shape_; }

    T* data() { return data_; }

    Tensor<T> at(const size_t& index) const
    {
        if (index >= this->shape_[0]) {
            throw out_of_range("Index out of range.");
        }
        size_t new_size = this->size_ / this->shape_[0];
        return Tensor(vector<size_t>(this->shape_.begin() + 1, this->shape_.end()),
                      new_size,
                      this->data_ + (index * new_size));
    }

    T& operator[](const size_t& index) const
    {
        if (index >= this->shape_[0]) {
            throw out_of_range("Index out of range.");
        }
        if (this->shape_.size() > 1) {
            throw length_error("Trying to access scalar value but array is multi-dimensional.");
        }
        return this->data_[index];
    }

    operator T() const { return this->data_[0]; }
    operator T&() { return this->data_[0]; }

    void operator=(vector<T> v)
    {
        if (v.size() != this->shape_[0] || this->shape_.size() > 1) {
            throw out_of_range("Wrong size for initialization from vector.");
        }
        copy(v.begin(), v.end(), this->data_);
    }

    void operator=(T t)
    {
        if (this->size_ != 1) {
            throw length_error("Trying to assign scalar value to multi-dimensional (>= 1) tensor.");
        }
        data_[0] = t;
    }

    void print() const
    {
        cout << "[";
        for (size_t i = 0; i < this->size_; ++i) {
            cout << this->data_[i];
            if (i != this->size_ - 1) {
                cout << ", ";
            }
        }
        cout << "]\n";
    }

    void print_shape() const
    {
        cout << "[";
        for (size_t i = 0; i < shape_.size(); ++i) {
            cout << shape_[i];
            if (i != shape_.size() - 1) {
                cout << ", ";
            }
        }
        cout << "]\n";
    }

    bool empty() const { return this->size_ == 0; }

    /*
     * Careful : after resize, all subarrays are not valid anymore !
     */
    void resize(const size_t& new_size)
    {
        if (new_size > size_)
            throw length_error("Invalid resize : new size should be smaller than old size.");
        size_t row_size = size_ / shape_[0];
        T* old_data = data_;
        T* new_data = new T[new_size * row_size];

        memcpy(new_data, old_data, new_size * row_size * sizeof(T));

        this->size_ = new_size * row_size;
        this->shape_[0] = new_size;
        this->data_ = new_data;
        delete[] old_data;
    }

    size_t total_size() const { return this->size_; }

    void copy_data(Tensor& other) const { memcpy(other.data(), data_, size_ * sizeof(T)); }

    T sum() const
    {
        T x;
        for (size_t i = 0; i < size_; i++)
            x += data_[i];
        return x;
    }

    void add_dimension()
    {
        vector<size_t> new_shape = vector<size_t>(shape_.size() + 1);
        new_shape[0] = 1;
        copy(shape_.begin(), shape_.end(), new_shape.begin() + 1);
        shape_ = new_shape;
    }

    void reset_data() { memset(data_, 0, size_ * sizeof(T)); }

  private:
    vector<size_t> shape_;
    size_t size_;
    T* data_;
    bool owner_;

    Tensor(const vector<size_t>& shape, const size_t& size, T* const data)
      : shape_(shape)
      , size_(size)
      , data_(data)
      , owner_(false)
    {
    }
};

#endif // __TENSOR_HPP__