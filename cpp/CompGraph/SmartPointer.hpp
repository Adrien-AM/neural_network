#ifndef __SMART_POINTER_HPP__
#define __SMART_POINTER_HPP__

template<typename T>
class SmartPointer
{
  private:
    T* ptr;
    unsigned int* cpt;

  public:
    SmartPointer()
      : ptr(nullptr)
      , cpt(nullptr)
    {
    }

    SmartPointer(T* p)
      : ptr(p)
      , cpt(new unsigned int(1))
    {
    }

    // Copy constructor
    SmartPointer(const SmartPointer& other)
      : ptr(other.ptr)
      , cpt(other.cpt)
    {
        (*cpt)++;
    }

    // Move Constructor
    SmartPointer(SmartPointer&& other) noexcept
      : ptr(other.ptr)
      , cpt(other.cpt)
    {
        other.ptr = nullptr;
        other.cpt = nullptr;
    }

    // Move Assignment Operator
    SmartPointer& operator=(SmartPointer&& other) noexcept
    {
        if (this != &other) {
            // Release the existing resources of *this
            if (ptr) {
                (*cpt)--;
                if (*cpt == 0) {
                    delete ptr;
                    delete cpt;
                }
            }
            // Acquire the resources of other
            ptr = other.ptr;
            cpt = other.cpt;
            // Leave other in a valid but unspecified state
            other.ptr = nullptr;
            other.cpt = nullptr;
        }
        return *this;
    }

    bool operator==(const SmartPointer<T>& other) const
    {
        return other.ptr == this->ptr;
    }

    bool operator!=(const SmartPointer<T>& other) const
    {
        return other.ptr != this->ptr;
    }

    SmartPointer<T>& operator++() {
        ptr++;
        return *this;
    }

    // Destructor
    ~SmartPointer()
    {
        if (ptr) {
            (*cpt)--;
            if (*cpt == 0) {
                delete ptr;
                delete cpt;
            }
            ptr = nullptr;
            cpt = nullptr;
        }
    }

    // Overload * operator to access the pointed object
    T& operator*() { return *ptr; }

    // Overload -> operator to access the members of the pointed object
    T* operator->() { return ptr; }

    // Overload assignment operator
    SmartPointer& operator=(const SmartPointer& other)
    {
        if (this == &other) {
            return *this; // Self-assignment check
        }

        // Decrement the reference count for the current object
        if (ptr) {
            (*cpt)--;
            if (*cpt == 0) {
                delete ptr;
                delete cpt;
            }
        }

        // Copy the values from the other object
        ptr = other.ptr;
        cpt = other.cpt;
        (*cpt)++;

        return *this;
    }
};
#endif // __SMART_POINTER_HPP__