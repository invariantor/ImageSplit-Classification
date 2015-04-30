import numpy as np
import pylab
import mahotas as mh
import types

# constants
upper_distance = 100  #the start searching 
approxWidth = 40
threshold = 300
border = 1

def pre_process(image):
    """
    
    pre_process will return black_white image, given a colorful image as input.        
    """
    T = mh.thresholding.otsu(image)
    image1 =image > T
    image2 = [[0]* image1.shape[1] for i in range(image1.shape[0])] 
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            if (image1[i][j] != [0,0,0]).any():
                image2[i][j] = 1                        
    image2 = np.array(image2, dtype = np.uint8)    
    return image2

def locate(image):
    """ 
    
    Given an screenshot as input, return the position of the matching game
    as well as the size of the game(num_x,num_y) 
    and the size of each grids(size_x,size_y).        
    """
    image = pre_process(image)    
    height,width = image.shape
    
    # stop going down when a grid is found
    up = upper_distance
    while True:
        num_white =0
        for j in range(width):
            num_white+=image[up][j]        
        if num_white>(approxWidth/2):
            break
        up +=1
    
    # stop going up when a grid is found
    down = height-1
    pre_num_white =0        #the number of white pixels in the last step 
    for j in range(width):
        pre_num_white+=image[down][j]
    while True:
        num_white =0
        for j in range(width):
            num_white+=image[down][j]
        
        if num_white-pre_num_white>(approxWidth/2):
            break
        pre_num_white = num_white
        down -=1
                
    current_image = image[up:]          
    """cut the top part(including the time bar, all sorts of buttons) away 
    which will interfere with our searching process"""    
    current_image = np.array(current_image)
    c_height,c_width = current_image.shape
    
    # stop going right when a grid is found    
    left = 0
    pre_num_white =0
    for i in range(c_height):
        pre_num_white+=current_image[i][left]
    while True:
        num_white =0
        for i in range(c_height):
            num_white+=current_image[i][left]
        
        if num_white-pre_num_white>(approxWidth/2):
            break
        pre_num_white = num_white
        left +=1
        
    # stop going left when a grid is found    
    right = c_width-1
    pre_num_white =0
    for i in range(c_height):
        pre_num_white+=current_image[i][right]
    while True:
        num_white =0
        for i in range(c_height):
            num_white+=current_image[i][right]
        
        if num_white-pre_num_white>(approxWidth/2):
            break
        pre_num_white = num_white
        right -=1
    
    
    temp = [0]*(down+1-up)
    for i in range(len(temp)):
        temp[i] = current_image[i][left:right+1]
        
    current_image = np.array(temp)
    height,width = current_image.shape
    
   
    divd_x = []
    for i in range(height):
        num_white = sum(current_image[i])
        if num_white < approxWidth/2:
            divd_x.append(i)
    temp_x = [divd_x[i] for i in range(len(divd_x)) if ((i==0) or (i==len(divd_x)-1)) or not (divd_x[i-1]+1==divd_x[i] and divd_x[i+1]-1==divd_x[i])]
    # only keep the truly dividing lines, namely  those marginal lines. 
    divd_x =temp_x
    
    divd_y = []
    for j in range(width):
        num_white = 0
        for i in range(height):
            num_white += current_image[i][j]
        if num_white < approxWidth/2:
            divd_y.append(j)
    temp_y = [divd_y[i] for i in range(len(divd_y)) if ((i==0) or (i==len(divd_y)-1)) or not (divd_y[i-1]+1==divd_y[i] and divd_y[i+1]-1==divd_y[i])]
    # only keep the truly dividing lines, namely  those marginal lines. 
    divd_y = temp_y
    
    #print divd_x
    #print divd_y
    
    """ 
    This part needs further refinement.
    """
    if len(divd_x):
        size_x = divd_x[0]
        num_x = divd_x[-1] / size_x +1
    else:
        size_x = height - 1
        num_x = 1
        
    if len(divd_y):
        size_y = divd_y[0]
        num_y = divd_y[-1] / size_y +1
    else:
        size_y = height - 1
        num_y = 1
        
    position = (up,down,left,right)
    info = (size_x,size_y,num_x,num_y)
    
    return  position, info 
    
    
    
def split(image,position,info):
    """
    
    Return a 2d matrix label, which labels different kinds of grids using natural numbers.
    (By convention, the empty grid is labeled 0)
        
    """
    size_x, size_y, num_x, num_y = info
    up, down, left, right = position
    
    T = mh.thresholding.otsu(image)
    image = image >T
    temp = [0]* (down+1-up)
    for i in range(len(temp)):
        temp[i] = image[up+i][left:right+1]
    temp = np.array(temp)
    image = temp
    
   
    game = [[0]* num_y for j in range(num_x)]
    for i in range(num_x):
        for j in range(num_y):
            grid = [0]* size_x
            for k in range(size_x):
                grid[k] = image[i*(size_x+1)+k][j*(size_y+1):(j+1)*(size_y+1)-1]    
            game[i][j] = grid
    
    # using a quite naive method -- calculating the statistical distance between two grids
    # improvement is needed here, to speed up the program
    black = [[[0]*3]*size_y]*size_x
    records = [black]
    label = [[0]* num_y for j in range(num_x)]
    for i in range(num_x):
        for j in range(num_y):
            find = False
            for index in range(len(records)):
                if distance(records[index],game[i][j])< threshold:
                    label[i][j] = index
                    find =True
                    break
            if not find:            
                records.append(game[i][j])
                label[i][j] = len(records)-1
    return label
    
def distance(a1,a2):
    """
    
    recursively calculate the distance between a1 and a2 
    """
    if (type(a1)== np.uint8) or (type(a1) == types.IntType) or (type(a1)==np.bool_):
        return abs(int(a1)-int(a2)) 
    if len(a1)!= len(a2): 
        print "Wrong Format","len(a1)=",len(a1),"len(a2)=",len(a2)
        return    
    dis =0
    for i in range(len(a1)):
        dis += distance(a1[i],a2[i])
    return dis