import os
os.environ['NUMBAPRO_LIBDEVICE'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\nvvm\libdevice"
os.environ['NUMBAPRO_NVVM'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\nvvm\bin\nvvm64_40_0.dll"

import matplotlib.pyplot as plt
from numba import cuda
import numpy as np
import time
from PIL import Image

#percentileFile = open("percentile.txt", "w")
#percentileData = open("percentile.txt", "r")
#iterationData = percentileData.readline()

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

@cuda.jit(device=True)
def burningShip(x, y, maxIterations):
    startingPoint = complex(x, y)
    #z = complex(-1, 0.33)
    z = 0
    for iteration in range(maxIterations):
        #z = (zz) + startingPoint

        shipZ = complex(abs(z.real), -abs(z.imag))
        z = (shipZ * shipZ) + startingPoint

        if ((z.real* z.real) + (z.imag*z.imag)) >= 4:
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

    #iterations1 = burningShip(real, imag, max_iters)
    #iterations2 = burningShip(real2, imag2, max_iters)

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
     #colorPoints = [(0, (255, 255, 255)), (0.2,(255,204,0)), (0.4,(135,30,20)), (0.6,(0,0,153)), (0.8,(0,0,153)), (1, (255, 255, 255))] # earth and sky
     #colorPoints = [(0, (255, 255, 255)), (0.2,(0,102,255)), (0.8,(255,0,204)), (1, (255, 255, 255))] #pink to blue
     #colorPoints = [(1/6, (250,235,44)), (2/6, (245,39,137)), (3/6, (233,0,255)), (4/6, (22,133,248)), (5/6, (61,20,76)), (6/6, (250,235,44))] # neon 80s
     #colorPoints = [(1/6, (113,28,145)), (2/6, 	(234,0,217)), (3/6,	(10,189,198)), (4/6, (19,62,124)), (5/6, (9,24,51)), (6/6, (113,28,145))] # cyberpunk neon
     colorPoints = [(1/6, (97,179,255)), (2/6, 	(33,10,127)), (3/6,	(5,136,218)), (4/6, (11,204,49)), (5/6, (33,253,43)), (6/6, (97,179,255))] # blue-green
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
    
    #Percentile1 = int(np.percentile(positiveFlattenedData, 1))
    #Percentile5 = int(np.percentile(positiveFlattenedData, 5))
    #Percentile50 = int(np.percentile(positiveFlattenedData, 50))
    #Percentile95 = int(np.percentile(positiveFlattenedData, 95))
    #Percentile99 = int(np.percentile(positiveFlattenedData, 99))
    Percentile995 = int(np.percentile(positiveFlattenedData, 99.5))

    #largestValue = int(max(positiveFlattenedData))

    #print(f"\n 1%: {Percentile1} \n 5%: {Percentile5} \n 50%: {Percentile50} \n 95%: {Percentile95} \n 99%: {Percentile99} \n 99.9%: {Percentile999} \n Largest Value: {largestValue} \n")
    #print(Percentile99)
    return Percentile995

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

centreX = (0.3491282064524898664630 + 0.3491282064602419514630) /2
centreY = (0.0654137641785317986460 + 0.0654137641843458623960)/ 2

centreX = (0.2537406773317819555850 + 0.2537406773340697113470)/2
centreY = (0.0003653040902033367730 + 0.0003653040914901993890)/2

centreX = (-1.5019831037155986726550 + -1.5019831037155878820160)/2 #used in blue-pink zoom
centreY = (-0.0022347151616106755922 + -0.0022347151616025826130)/2

centreX = (-0.386270986024843662762 + -0.386270986023386033294)/2 # retro neon zoom/cyberpunk
centreY = (0.623332316261392914773 + 0.623332316262212831349)/2


centreX = -1.40115569 # leaves
centreY = 0
'''
centreX = (-0.7499025744650928849191 + -0.7499025744650581748143)/2 # Seahorse valley zoom
centreY = (0.0168320487928356486357 + 0.0168320487928616812145)/2


span = 100

maxIterationsFloat = 20
maxIterations = int(maxIterationsFloat)
gradientRepeats = 1.4
requiredLength = 4 #for name of file output

gradientLookup = {}

imageNumber = 0

for imageNumber in range(9999) :

    startTime = time.perf_counter()

    span *= 0.9965
    #maxIterations  = iterationsCalculator(iterationData, imageNumber)
    #print(f"Frame {imageNumber}, iterations: {maxIterations}")

    renderedImage = Image.new('RGB', (xRes,yRes))
    #L for greyscale, RGB for RGB, HSV for HSV

    xMin, xMax, yMin, yMax = visibleArea(span, centreX, centreY)

    pixelData = calculateMandelbrot(xRes, yRes, maxIterations, xMin, xMax, yMin, yMax)

    
    #analysis fuctions \/
    Percentile995 = iterationAnalysis(pixelData)

    if (maxIterationsFloat - Percentile995)/maxIterationsFloat < 0.35:
        maxIterationsFloat = maxIterationsFloat * 1.005


    maxIterations = round(maxIterationsFloat)
    #percentileFile.write(str(Percentile99)+"\n")
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
    
    renderedImage.save(f"Seahorse{zeroFill(imageNumber, requiredLength)}.png")

    
    endTime = time.perf_counter()
    print(f"Frame {imageNumber} took {round((endTime-startTime),1)}s and did {maxIterations} iterations, the 99.5 percentile was {Percentile995}, {round(((Percentile995/maxIterations)*100), 0)}% of maximum iterations")
