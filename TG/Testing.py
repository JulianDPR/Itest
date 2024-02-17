import time
import multiprocessing
import threading

# Define the function representing the CPU-bound task
def square_numbers(numbers):
    return [x ** 2 for x in numbers]

if __name__ == "__main__":
    # Generate a large list of numbers
    numbers = list(range(1, 1000001))  # 1 million numbers

    # Speed test using multiprocessing
    start_time = time.time()
    with multiprocessing.Pool(12) as pool:
        result_multiprocessing = pool.map(square_numbers, [numbers])[0]
    end_time_multiprocessing = time.time()
    multiprocessing_time = end_time_multiprocessing - start_time

    # Speed test using threading
    start_time = time.time()
    thread = threading.Thread(target=square_numbers, args=(numbers,))
    thread.start()
    thread.join()
    end_time_threading = time.time()
    threading_time = end_time_threading - start_time

    # Print the results
    print("Time taken using multiprocessing:", multiprocessing_time)
    print("Time taken using threading:", threading_time)