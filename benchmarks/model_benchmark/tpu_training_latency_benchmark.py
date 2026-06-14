"""Benchmark script to validate and simulate Keras 3
TPU training execution latency."""

import time
from typing import Dict
from typing import List

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags

import keras
from keras import layers

FLAGS = flags.FLAGS
flags.DEFINE_integer("steps_per_execution", 64, "Steps per execution.")
flags.DEFINE_integer("batch_size", 32, "Training batch size.")
flags.DEFINE_integer("num_steps", 1000, "Total number of steps to run.")
flags.DEFINE_integer("num_inputs", 20, "Number of distinct input tensors.")
flags.DEFINE_boolean(
    "jit_compile", False, "Whether to enable XLA JIT compilation."
)


class StepLatencyCallback(keras.callbacks.Callback):
    """Callback to measure exact step and epoch latency."""

    def __init__(self, warmup_steps: int = 100):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.start_time = 0.0
        self.step_times: List[float] = []

    def on_train_batch_begin(self, batch: int, logs: Dict[str, float] = None):
        if batch >= self.warmup_steps:
            self.start_time = time.time()

    def on_train_batch_end(self, batch: int, logs: Dict[str, float] = None):
        if batch >= self.warmup_steps and self.start_time > 0:
            self.step_times.append(time.time() - self.start_time)


def build_multi_input_model(num_inputs: int) -> keras.Model:
    """Builds a multi-input model to simulate heavy merge/concat operations."""
    inputs = []
    processed = []
    for i in range(num_inputs):
        inp = keras.Input(shape=(16,), name=f"input_{i}")
        inputs.append(inp)
        # Apply a dense projection
        x = layers.Dense(16, activation="relu")(inp)
        processed.append(x)

    # Concatenate all features alongside axis -1
    merged = layers.Concatenate(axis=-1)(processed)
    # Deep prediction head
    x = layers.Dense(128, activation="relu")(merged)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs=inputs, outputs=outputs)


def run_benchmark() -> None:
    """Runs the execution latency benchmark and outputs average throughput."""
    keras.config.disable_traceback_filtering()
    print(
        f"--- Running Validation Benchmark with "
        f"steps_per_execution={FLAGS.steps_per_execution}, "
        f"jit={FLAGS.jit_compile} ---"
    )
    model = build_multi_input_model(FLAGS.num_inputs)

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = keras.losses.BinaryCrossentropy()

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        steps_per_execution=FLAGS.steps_per_execution,
        jit_compile=FLAGS.jit_compile,
    )

    # Create dummy dataset
    def gen_dummy_data():
        while True:
            xs = {
                f"input_{i}": np.random.randn(FLAGS.batch_size, 16).astype(
                    "float32"
                )
                for i in range(FLAGS.num_inputs)
            }
            y = np.random.randint(0, 2, size=(FLAGS.batch_size, 1)).astype(
                "float32"
            )
            yield xs, y

    dataset = tf.data.Dataset.from_generator(
        gen_dummy_data,
        output_signature=(
            {
                f"input_{i}": tf.TensorSpec(
                    shape=(FLAGS.batch_size, 16), dtype=tf.float32
                )
                for i in range(FLAGS.num_inputs)
            },
            tf.TensorSpec(shape=(FLAGS.batch_size, 1), dtype=tf.float32),
        ),
    ).prefetch(tf.data.AUTOTUNE)

    timer_cb = StepLatencyCallback(warmup_steps=FLAGS.steps_per_execution * 2)

    start_wall_time = time.time()
    model.fit(
        dataset,
        steps_per_epoch=FLAGS.num_steps,
        callbacks=[timer_cb],
        verbose=1,
    )
    total_wall_time = time.time() - start_wall_time

    # Calculate metrics
    measured_steps = len(timer_cb.step_times)
    if measured_steps > 0:
        avg_step_time_ms = (sum(timer_cb.step_times) / measured_steps) * 1000.0
        throughput = 1.0 / (sum(timer_cb.step_times) / measured_steps)
        print("\n" + "=" * 50)
        print("📊 BENCHMARK EXECUTION SUMMARY")
        print("=" * 50)
        print(f"Total Steps Executed : {FLAGS.num_steps}")
        print(f"Steps Per Execution  : {FLAGS.steps_per_execution}")
        print(f"XLA JIT Compilation  : {FLAGS.jit_compile}")
        print(f"Total Wall-clock Time: {total_wall_time:.2f} seconds")
        print(f"Average Step Latency : {avg_step_time_ms:.2f} ms")
        print(f"Throughput           : {throughput:.2f} steps/sec")
        print("=" * 50 + "\n")
    else:
        print(
            "Warning: Could not capture step times. Ensure "
            "num_steps > warmup_steps."
        )


def main(_) -> None:
    run_benchmark()


if __name__ == "__main__":
    app.run(main)
