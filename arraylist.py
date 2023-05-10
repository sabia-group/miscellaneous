# Define a numpy object with a method "update"
# which is analagous to a list "append".

class arraylist:
    def __init__(self):
        self.data = np.zeros((100000,))
        self.capacity = 100000
        self.size = 0

    def update(self, row):
      # The argument row must be a numpy object in this implementation
        n = row.shape[0]
        self.add(row,n)

    def add(self, x, n):
      # Increase the size of the numpy object if necessary.
      # Then append the new data.
        if self.size+n >= self.capacity:
            self.capacity *= 2
            newdata = np.zeros((self.capacity,))
            newdata[:self.size] = self.data[:self.size]
            self.data = newdata

        self.data[self.size:self.size+n] = x
        self.size += n

    def finalize(self):
      # Return the 1D array removing any unnecessary zeros.
        return self.data[:self.size]
