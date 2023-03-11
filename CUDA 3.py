import os
os.environ['NUMBAPRO_LIBDEVICE'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\nvvm\libdevice"
os.environ['NUMBAPRO_NVVM'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\nvvm\bin\nvvm64_40_0.dll"

import matplotlib.pyplot as plt
from numba import cuda
import numpy as np
import time
from PIL import Image

#percentileFile = open("percentile.txt", "w")
percentileData = open("percentile.txt", "r")
iterationData = percentileData.readline()

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
def mandelbrot(x, y, maxIterations):
    startingPoint = complex(x, y)
    z = startingPoint
    for iteration in range(maxIterations):
        z = (z*z) + startingPoint
        if (z.real ** 2 + z.imag ** 2) >= 4:
             return iteration
    return -1


@cuda.jit
def fillArray(start_x, start_y, pixel_x, pixel_y, width, height, max_iters, array):
  x, y = cuda.grid(2)
  if x < width and y < height:
    real = start_x + x * pixel_x
    imag = start_y + y * pixel_y

    real2 = start_x + (x+0.5) * pixel_x
    imag2 = start_y  + (y+0.5) * pixel_y

    iterations1 = mandelbrot(real, imag, max_iters)
    iterations2 = mandelbrot(real2, imag2, max_iters)


    averageIterations= (iterations1 + iterations2)/2

    if iterations1 == -1:
        averageIterations = iterations2
    if iterations2 == -1:
        averageIterations = iterations1

    array[height-y-1, x] = averageIterations

def calculateMandelbrot(width, height, max_iters, min_x, max_x, min_y, max_y):  
    threads = 16
    
    blocksX = width // threads + 1
    blocksY = height // threads + 1

    pixel_x = (max_x - min_x) / width
    pixel_y = (max_y - min_y) / height

    data = np.zeros((height, width))
  
    #startTime = time.perf_counter()
  
    d_data = cuda.to_device(data)
  
    fillArray[(blocksX, blocksY), (threads, threads)](min_x, min_y, pixel_x, pixel_y, width, height, max_iters, d_data)
    data = d_data.copy_to_host()
  
    #endTime = time.perf_counter()
  
    #print(f"Time to Calculate: {(endTime - startTime)}" )

    return data

def calculateGradient(positionAlongGradient):
     colorPoints = [(0, (255, 255, 255)), (0.2,(255,204,0)), (0.4,(135,30,20)), (0.6,(0,0,153)), (0.8,(0,0,153)), (1, (255, 255, 255))]
     #colorPoints are in the format:  (position, (r, g, b))

     i = 0
     while colorPoints[i][0] <= positionAlongGradient:
        i += 1

     topColour = colorPoints[i][1]
     bottomColour = colorPoints[i-1][1]

     difference = (topColour[0]-bottomColour[0], topColour[1]-bottomColour[1], topColour[2]-bottomColour[2])

     posBetweenColours = (positionAlongGradient - colorPoints[i-1][0]) / (colorPoints[i][0] - colorPoints[i-1][0])

     r = int(bottomColour[0] + posBetweenColours / 1 * difference[0])
     g = int(bottomColour[1] + posBetweenColours / 1 * difference[1])
     b = int(bottomColour[2] + posBetweenColours / 1 * difference[2])

     return (r, g, b)

def iterationAnalysis(pixelData):
    #pixel data comes in as a 2D array of iterations
    flattenedData = list(np.concatenate(pixelData).flat)

    positiveFlattenedData = [item for item in flattenedData if item >= 0]
    
    Percentile1 = int(np.percentile(positiveFlattenedData, 1))
    Percentile5 = int(np.percentile(positiveFlattenedData, 5))
    Percentile50 = int(np.percentile(positiveFlattenedData, 50))
    Percentile95 = int(np.percentile(positiveFlattenedData, 95))
    Percentile99 = int(np.percentile(positiveFlattenedData, 99))
    Percentile999 = int(np.percentile(positiveFlattenedData, 99.9))

    largestValue = int(max(positiveFlattenedData))

    #print(f"\n 1%: {Percentile1} \n 5%: {Percentile5} \n 50%: {Percentile50} \n 95%: {Percentile95} \n 99%: {Percentile99} \n 99.9%: {Percentile999} \n Largest Value: {largestValue} \n")
    #print(Percentile99)
    return Percentile99

def histogramPlotter(pixelData):

    flattenedData = list(np.concatenate(pixelData).flat)
    positiveFlattenedData = list([item for item in flattenedData if item >= 0])

    plt.hist(positiveFlattenedData, bins = 1000)

    plt.show()

def iterationsCalculator(iterationData, imageNumber):
    iterationData.replace("'", "")
    valuesList = list(iterationData.split(" "))
    newValuesList = []
    for value in valuesList:
        if value != "":
            newValuesList.append(int(value))
    #print(valuesList)
    endValue = int(int(imageNumber) + int(120))
    sumValues = sum(newValuesList[imageNumber : endValue])
    averageValue = sumValues/120
    return int(averageValue * 2)



#Resolution of output image
xRes = 2560
yRes = 1440
'''
centreX = (-1.186930005248785669013 + -1.186930005237193348409) /2
centreY = (0.300290948601785329546 + 0.300290948610505216726)/ 2
'''

centreX = (0.3491282064524898664630 + 0.3491282064602419514630) /2
centreY = (0.0654137641785317986460 + 0.0654137641843458623960)/ 2

span = 100 
#maxIterations = 20000
gradientRepeats = 1

requiredLength = 4 #for name of file output

gradientLookup = {}

imageNumber = 0

for imageNumber in range(5250) :
    startTime = time.perf_counter()

    span *= 0.995
    maxIterations  = iterationsCalculator(iterationData, imageNumber)
    print(f"Frame {imageNumber}, iterations: {maxIterations}")

    renderedImage = Image.new('RGB', (xRes,yRes))
    #L for greyscale, RGB for RGB, HSV for HSV

    xMin, xMax, yMin, yMax = visibleArea(span, centreX, centreY)

    pixelData = calculateMandelbrot(xRes, yRes, maxIterations, xMin, xMax, yMin, yMax)

    #analysis fuctions \/
    #Percentile99 = iterationAnalysis(pixelData)
    #percentileFile.write(str(Percentile99)+" ")
    #histogramPlotter(pixelData)

    
    for xIterations in range(xRes-1):
        for yIterations in range(yRes-1):
            pixelIterations = pixelData[yIterations,xIterations]

            if pixelIterations == -1:
                pixelColour = (0,0,0)
            else:
                 positionAlongGradient = ((pixelIterations/maxIterations)*gradientRepeats) % 1

                 if positionAlongGradient not in gradientLookup:
                     gradientLookup[positionAlongGradient] = calculateGradient(positionAlongGradient)
                     
                 pixelColour = gradientLookup[positionAlongGradient]
            
            renderedImage.putpixel((xIterations,yIterations),pixelColour)
    
    renderedImage.save(f"CUDA full zoom{zeroFill(imageNumber, requiredLength)}.png")


    endTime = time.perf_counter()
    print(f"Frame time: {(endTime-startTime)}")

