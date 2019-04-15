import numpy as np
def sample(img, spacing=20, pattern="cross", index=True):
    """
    ~~Yasin~~
    Samples a given flow simulation to get the head measurements:
    image:

    :spacing:            Desired spacing between the measuement points
    :param pattern:      Sampling scheme (cross, plus, square)
            index:       (boolean) Whether to return to the index values
                         of the samples or the head values

    :return:            (vector) It returns to the index values or the head values
    """
    # Calculate the number of points in X direction
    pointNum = int(img.nx / spacing - 1)
    # Create an array to store the indice values of the points
    ind = np.zeros((pointNum * 2, 2), dtype="int")  #
    # Get the head values at the chosen point locations
    for i in range(pointNum):
        ind[i] = ((i + 1) * spacing - 1), ((i + 1) * spacing - 1)
        ind[i + pointNum] = ((i + 1) * spacing - 1), (pointNum * spacing - 1 - i * spacing)
    # return to either index or head values based on the index=True or False
    if index:
        return ind
    else:
        values = img.val[:, 0, ind[:, 0], ind[:, 1]]
        return values