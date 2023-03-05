import os
os.environ['NUMBAPRO_LIBDEVICE'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\nvvm\libdevice"
os.environ['NUMBAPRO_NVVM'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\nvvm\bin\nvvm64_40_0.dll"

import numba
from numba import cuda
import numpy as np
import time
from PIL import Image



def visibleArea(span, centreX, centreY):
    spanX = span/9
    spanY = span/16

    xMin = centreX - (spanX/2)
    xMax = centreX + (spanX/2)

    yMin = centreY - (spanY/2)
    yMax = centreY + (spanY/2)

    return xMin, xMax, yMin, yMax

def zeroFill(inputNumber, requiredLength):
    number = str(inputNumber)
    numberLength = len(number)
    while numberLength < requiredLength:
        number = "0" + number
        numberLength = len(number)

    return str(number)

@cuda.jit(device=True)
def mandelbrot(x, y, max_iters):
    startingPoint = complex(x, y)
    z = startingPoint
    for iteration in range(max_iters):
        z = (z*z) + startingPoint
        if (z.real ** 2 + z.imag ** 2) >= 4:
            return iteration
    return 0

@cuda.jit
def fillArray(start_x, start_y, pixel_x, pixel_y, width, height, max_iters, array):
  x, y = cuda.grid(2)
  if x < width and y < height:
    real = start_x + x * pixel_x
    imag = start_y + y * pixel_y

    color = mandelbrot(real, imag, max_iters)
    array[height-y-1, x] = color

def calculateMandelbrot(width, height, max_iters, min_x, max_x, min_y, max_y):
  pixels = width * height
  
  threads = 16
    
  blocksX = width // threads + 1
  blocksY = height // threads + 1

  pixel_x = (max_x - min_x) / width
  pixel_y = (max_y - min_y) / height

  data = np.zeros((height, width))
  
  startTime = time.perf_counter()
  
  d_data = cuda.to_device(data)
  
  fillArray[(blocksX, blocksY), (threads, threads)](min_x, min_y, pixel_x, pixel_y, width, height, max_iters, d_data)
  data = d_data.copy_to_host()
  
  endTime = time.perf_counter()
  
  print(f"Time to Calculate: {(endTime - startTime)}" )

  return data
  

#Resolution of output image
xRes = 2560
yRes = 1440

centreX = -0.75
centreY = 0

span = 50
maxIterations = 15000

requiredLength = 4


for image in range(39):
    span *= 0.5

    renderedImage = Image.new('L', (xRes,yRes))
    #L for greyscale, RGB for RGB, HSV for HSV

    xMin, xMax, yMin, yMax = visibleArea(span, centreX, centreY)

    pixelData = calculateMandelbrot(xRes, yRes, maxIterations, xMin, xMax, yMin, yMax)

    for xIterations in range(xRes-1):
            for yIterations in range(yRes-1):
                recursions = pixelData[yIterations,xIterations]

                scaledRecursions = recursions ** 0.7205

                colourStepping = int(scaledRecursions % 255)

                #print(colourStepping)
                
                renderedImage.putpixel((xIterations,yIterations),colourStepping)
                
    renderedImage.save(f"Cuda test{zeroFill(image, requiredLength)}.png")


#calculateMandelbrot(2560, 1440, 50000, -1.186930005248785669013, -1.186930005237193348409, 0.300290948601785329546, 0.300290948610505216726)
















