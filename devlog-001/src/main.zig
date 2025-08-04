const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
var prng = std.Random.DefaultPrng.init(42);

const alpha: f64 = 0.5;

// Input is 784 x 1
// vectorized: 784 x 10,000

fn random_vector(comptime m: usize) [m]f64 {
  var output: [m]f64 = undefined;
  for (output, 0..) |_, idx| {
    output[idx] = prng.random().float(f64) * 2 - 1;
  }
  return output;
}

fn random_matrix(comptime m: usize, comptime n: usize) [m][n]f64 {
  var output: [m][n]f64 = undefined;
  for (output, 0..) |row, row_idx| {
    for (row, 0..) |_, col_idx| {
      output[row_idx][col_idx] = (prng.random().float(f64) * 2 - 1) * 0.1;
    }
  }
  return output;
}

fn one_hot(label: u8) [10]f64 {
  var output: [10]f64 = .{0.0} ** 10;
  output[label] = 1.0;
  return output;
}

pub fn ReLU(comptime T: type, comptime m: usize, x: [m]T) [m]T {
  var output: [m]T = undefined;
  for (x, 0..) |row, idx| {
    output[idx] = @max(0, row);
  }
  return output;
}

pub fn ReLU_deriv(comptime T: type, x: T) T {
  return if (x < 0) @as(T, 0) else @as(T, 1);
}

// Softmax
pub fn softmax(comptime T: type, comptime m: usize, v: [m]T) [m]T{
  var max: T = v[0];
  var output: [m]f64 = undefined;
  var sum: f64 = 0.0;

  // Get max value
  for (v[1..]) |val| {
    if (val > max) {
      max = val;
    }
  }

  // Find exponentials relative to max value (this prevents overflow)
  for (v, 0..) |val, idx| {
    output[idx] = @exp(val - max);
    sum += output[idx];
  }

  // Normalize
  for (&output) |*x| {
    x.* /= sum;
  }

  return output;
}

pub fn add(comptime T: type, comptime m: usize, v1: [m]T, v2: [m]T) [m]T {
  var output: [m]T = undefined;
  for (v1, v2, 0..) |x, y, idx| {
    output[idx] = x + y;
  }
  return output;
}

pub fn sub(comptime T: type, comptime m: usize, v1: [m]T, v2: [m]T) [m]T {
  var output: [m]T = undefined;
  for (v1, v2, 0..) |x, y, idx| {
    output[idx] = x - y;
  }
  return output;
}

pub fn vsum(comptime T: type, comptime m: usize, v: [m]T) T {
  var output: f64 = 0.0;
  for (v) |val| {
    output += val;
  }
  return output;
}

// Dot Product
pub fn dot(comptime T: type, comptime m: usize, comptime n: usize, mat1: [m][n]T, mat2: [n]T) [m]T {
  var output: [m]T = undefined;

  for (mat1, 0..) |row, row_idx| {
    for (row, mat2)  |col1, col2| {
      output[row_idx] += col1 * col2;
    }
  }

  return output;
}

pub fn vmax(comptime T: type, v: [10]T) usize {
  var output: usize = 0;
  var max: T = @as(T, 0.0);
  for (v, 0..) |i, idx| {
    if (i > max) {
      max = i;
      output = idx;
    }
  }
  return output;
}

const Image = struct {
  image: [784]f64,
  label: u8,
};

pub fn main() !void {
  ////////////////////////////////////////////////
  // Parse MNIST Training and Test Data //////////
  ////////////////////////////////////////////////
  var gpa = std.heap.GeneralPurposeAllocator(.{}){};
  defer _ = gpa.deinit();
  const alloc = gpa.allocator();

  const cwd = std.fs.cwd();

  const trainFileContents = try cwd.readFileAlloc(alloc, "./mnist/mnist_handwritten_train.json", 500_000_000);
  defer _ = alloc.free(trainFileContents);

  const testFileContents = try cwd.readFileAlloc(alloc, "./mnist/mnist_handwritten_test.json",74_000_000);
  defer _ = alloc.free(testFileContents);

  std.debug.print("////////// Parsing MNIST Training Data ///////////\n", .{});
  const data_train = try std.json.parseFromSlice([]Image, alloc, trainFileContents,.{});
  defer data_train.deinit();
  std.debug.print("/////////////////// Complete /////////////////////\n", .{});

  std.debug.print("//////////// Parsing MNIST Test Data //////////////\n", .{});
  const data_test = try std.json.parseFromSlice([]Image, alloc, testFileContents,.{});
  defer data_test.deinit();
  std.debug.print("/////////////////// Complete /////////////////////\n", .{});

  // Network Architecture
  // In     H1     Out
  // 784 -> 256 -> 10

  // Hidden layer 1
  // weights: 256 x 784
  // neurons: 256
  // bias: 256
  var w1: [256][784]f64 = random_matrix(256, 784);
  var b1: [256]f64 = random_vector(256);

  // Hidden layer 2 (output layer)
  // weights: 10 x 256
  // neurons: 10
  // bias: 10
  var w2: [10][256]f64 = random_matrix(10, 256);
  var b2: [10]f64 = random_vector(10);

  var z1: [256]f64 = undefined;
  var a1: [256]f64 = undefined;
  var z2: [10]f64 = undefined;
  var a2: [10]f64 = undefined;

  var right: f32 = 0;
  var wrong: f32 = 0;
  // Stochastic Gradient Descent
  // Only do one sample at a time
  // Note: It is far better to perform vectorization over the entire dataset
  for (0..10) |epoch| {

    std.debug.print("Epoch: {d}\n", .{epoch});
    for (data_train.value, 0..) |data, iteration| {
      // Input layer (Data)
      const input: [784]f64 = data.image;
      const label: u8 = data.label;

      ////////////////////////////////////////////////
      // Forward Propogation /////////////////////////
      ////////////////////////////////////////////////
      z1 = dot(f64, 256, 784, w1, input);
      z1 = add(f64, 256, z1, b1);
      a1 = ReLU(f64, 256, z1);

      z2 = dot(f64, 10, 256, w2, a1);
      z2 = add(f64, 10, z2, b2);
      a2 = softmax(f64, 10, z2);

      const guess = vmax(f64, a2);
      // Display Guess
      // std.debug.print("Input: {d}    ", .{label});
      // std.debug.print("Guess: {d}\n", .{guess});
      if (label == guess) {
        right += 1.0;
      } else {
        wrong += 1.0;
      }

      if (iteration % 100 == 0) {
        std.debug.print("Accuracy: {d:.6}\n", .{right / (right + wrong)});
      }


      ////////////////////////////////////////////////
      // Back Propogation ////////////////////////////
      ////////////////////////////////////////////////
      const y = one_hot(label);         // 10 x 1

      var dz2: [10]f64 = undefined; // 10 x 1
      for (0..10) |idx| {
        dz2[idx] = a2[idx] - y[idx];
      }

      // dw2 = dot(dz2, a1.transpose())
      var dw2: [10][256]f64 = undefined;
      for (dz2, 0..) |row, m| {       // 10 x 1
        for (a1, 0..) |col, n| {      // 256 x 1 transposed => 1 x 256
          dw2[m][n] = (1/60000) * row * col;
        }
      }
      const db2: f64 = (1/60000) * vsum(f64, 10, dz2);

      var dz1: [256]f64 = undefined;
      // dot( w2.T , dz2) * ReLU_deriv(z1);
      for (0..256) |i| {
        for (0..10) |j| {
          dz1[i] += w2[j][i] * dz2[j];
        }
        dz1[i] *= ReLU_deriv(f64, z1[i]);
      }

      var dw1: [256][784]f64 = undefined;
      for (dz1, 0..) |row, m| {       // 256 x 1
        for (input, 0..) |col, n| {   // 784 x 1 transposed => 1 x 784
          dw1[m][n] = (1/60000) * row * col;
        }
      }

      const db1: f64 = (1/60000) * vsum(f64, 256, dz1);

      ////////////////////////////////////////////////
      // Update Network //////////////////////////////
      ////////////////////////////////////////////////

      // Update w1
      for (w1, 0..) |row, m| {
        for (row, 0..) |_, n| {
          w1[m][n] -= alpha * dw1[m][n];
        }
      }

      // Update b1
      for (b1, 0..) |_, m| {
        b1[m] -= alpha * db1;
      }

      // Update w2
      for (w2, 0..) |row, m| {
        for (row, 0..) |_, n| {
          w2[m][n] -= alpha * dw2[m][n];
        }
      }

      // Update b2
      for (b2, 0..) |_, m| {
        b2[m] -= alpha * db2;
      }
    }
  }

}

test "one-hot" {
  const zero = one_hot(0);
  const five = one_hot(5);
  std.debug.print("one_hot(0) = {d}\n", .{zero});
  std.debug.print("one_hot(5) = {d}\n", .{five});
}

test "add vectors" {
  const v1: [5]i32 = .{1, 2, 3, 4, 5};
  const v2: [5]i32 = .{10, 20, 30, 40, 50};
  const vout = add(i32, 5, v1, v2);
  std.debug.print("v1 + v2 = {d}\n", .{vout});
}

test "sub vectors" {
  const v1: [5]i32 = .{10, 20, 30, 40, 50};
  const v2: [5]i32 = .{1, 2, 3, 4, 5};
  const vout = sub(i32, 5, v1, v2);
  std.debug.print("v1 + v2 = {d}\n", .{vout});
}

test "relu" {
  const z: [2]f64 = .{5.0, -5.0};
  std.debug.print("ReLU([5, -5]) = {d}\n", .{ReLU(f64, 2, z)});
}

test "relu derivative" {
  const x: f64 = 5.0;
  const y: f64 = -5.0;
  std.debug.print("ReLU'(5.0) = {d}\n", .{ReLU_deriv(f64, x)});
  std.debug.print("ReLU'(-5.0) = {d}\n", .{ReLU_deriv(f64, y)});
}

test "dot product" {
  const arry1: [2][3]f64 = .{
    .{1.5, 2.8, 4.2},
    .{2.2, 0.0, 4.9}
  };

  const arry2: [3]f64 = .{1.0, 1.0, 1.0};
  const output = dot(f64, 2, 3, arry1, arry2);

  for (output) |row| {
    std.debug.print("{d:.4}\n", .{row});
  }
}
