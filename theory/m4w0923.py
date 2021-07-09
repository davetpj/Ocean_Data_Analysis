import numpy as np
import matplotlib.pyplot as plt

input_data = "1,234,567"


def convert_int(input_data=input_data):
    output_data = int(input_data.replace(",", ""))
    return output_data


print(convert_int())
print(type(convert_int()))


# 2
samples = np.random.normal(0, 1, 10000)
plt.hist(samples, bins=100)
