from flask import Flask, request
from flask import render_template
import time
import json
from scipy.interpolate import interp1d
import numpy as np
import math
from sklearn.metrics.pairwise import euclidean_distances

app = Flask(__name__)

centroids_X = [50, 205, 135, 120, 100, 155, 190, 225, 275, 260, 295, 330, 275, 240, 310, 345, 30, 135, 85, 170, 240,
               170, 65, 100, 205, 65]
centroids_Y = [85, 120, 120, 85, 50, 85, 85, 85, 50, 85, 85, 85, 120, 120, 50, 50, 50, 50, 85, 50, 50, 120, 50, 120, 50,
               120]

words, probabilities = [], {}
template_points_X, template_points_Y = [], []
file = open('words_10000.txt')
content = file.read()
file.close()
content = content.split('\n')
for line in content:
    line = line.split('\t')
    words.append(line[0])
    probabilities[line[0]] = float(line[2])
    template_points_X.append([])
    template_points_Y.append([])
    for c in line[0]:
        template_points_X[-1].append(centroids_X[ord(c) - 97])
        template_points_Y[-1].append(centroids_Y[ord(c) - 97])


def generate_sample_points(points_X, points_Y):
    '''Generates 100 sampled points for a gesture.

    In this function, we convert every gesture or template to a set of 100 points, such that we can compare
    the input gesture and a template.

    :param points_X: A list of X-axis values of a gesture.
    :param points_Y: A list of Y-axis values of a gesture.

    :return:
        sample_points_X: A list of X-axis values of a gesture after sampling, containing 100 elements.
        sample_points_Y: A list of Y-axis values of a gesture after sampling, containing 100 elements.
    '''

    # Setting the list of x-axis and y-axis values of the gesture
    temp_x = points_X
    temp_y = points_Y
    # The differences between consecutive elements of an array.

    # np. ediff1d is used to calculate the euclidean distance between consecutive points
    # a = [10,100,500], np.ediff1d(a, to_begin=0) = [  0,  90, 400]
    # euc_dist = sqrt(x^2 + y^2)
    euc_dist = np.sqrt(np.ediff1d(points_X, to_begin=0) ** 2 + np.ediff1d(points_Y, to_begin=0) ** 2)

    # Calculate the cumulative distance
    eu_distance = np.add.accumulate(euc_dist)

    # x_points = [275, 275] and y_points = [50, 50]. So, Euclidean distance = 0, and thus I have added a check that if 0, then return the list of 100 points, with all elements as same value.
    # Here, sample_points_x = [275, 275, 275, ….] 100 times and sample_points_y = [50, 50, 50, ….] 100 times.
    if eu_distance[-1] == 0:
        return [points_X[0] for _ in range(100)], [points_Y[0] for _ in range(100)]

    # Find ratio, which will be the x-axis in the interpolated plot. the true value of point
    eu_distance = eu_distance / eu_distance[-1]

    # Now, I am carrying out interpolation. Referred https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
    inter_x, inter_y = interp1d(eu_distance, temp_x, kind='linear'), interp1d(eu_distance, temp_y, kind='linear')

    # We need divide the distance into 100 equidistant points, that would complete the process of sampling.
    alpha = np.linspace(0, 1, 100)

    return inter_x(alpha).tolist(), inter_y(alpha).tolist()


template_sample_points_X, template_sample_points_Y = [], []
for i in range(10000):
    X, Y = generate_sample_points(template_points_X[i], template_points_Y[i])
    template_sample_points_X.append(X)
    template_sample_points_Y.append(Y)


def do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y):
    '''Do pruning on the dictionary of 10000 words.

    In this function, we use the pruning method described in the paper (or any other method you consider reasonable)
    to narrow down the number of valid words so that ambiguity can be avoided.

    :param gesture_points_X: A list of X-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param gesture_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.

    :return:
        valid_words: A list of valid words after pruning.
        valid_probabilities: The corresponding probabilities of valid_words.
        valid_template_sample_points_X: 2D list, the corresponding X-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
        valid_template_sample_points_Y: 2D list, the corresponding Y-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
    '''
    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = [], [], []
    threshold = 30  # More the pruning threshold, more valid words will be formed

    """
    To filter out some templates, SHARK2 computes start-to-start and end-to-end distances between a template and the unknown gesture.
    """
    gesture_x_start, gesture_x_end = gesture_points_X[0][0], gesture_points_X[0][-1]
    gesture_y_start, gesture_y_end = gesture_points_Y[0][0], gesture_points_Y[0][-1]
    # Use math.sqrt as it is faster

    # as per paper, effective  filtering  mechanism  we  found is based on the start and end positions of the  sokgraph  templates,  normalized  in  scale  and  translation
    for idx, (sample_point_x, sample_point_y) in enumerate(zip(template_sample_points_X, template_sample_points_Y)):
        start_dist = math.sqrt(
            ((gesture_x_start - sample_point_x[0]) ** 2) + ((gesture_y_start - sample_point_y[0]) ** 2))
        end_dist = math.sqrt(((gesture_x_end - sample_point_x[-1]) ** 2) + ((gesture_y_end - sample_point_y[-1]) ** 2))
        if end_dist <= threshold and start_dist <= threshold:
            valid_template_sample_points_X.append(sample_point_x)
            valid_template_sample_points_Y.append(sample_point_y)
            valid_words.append(content[idx].split('\t')[0])

    return valid_words, valid_template_sample_points_X, valid_template_sample_points_Y


def get_scaled_points(sample_points_X, sample_points_Y, L):
    x_maximum = max(sample_points_X)
    x_minimum = min(sample_points_X)
    W = x_maximum - x_minimum
    y_maximum = max(sample_points_Y)
    y_minimum = min(sample_points_Y)
    H = y_maximum - y_minimum
    try:
        r = L / max(H, W)
    except ZeroDivisionError:
        r = 1

    gesture_X, gesture_Y = [], []
    for point_x, point_y in zip(sample_points_X, sample_points_Y):
        gesture_X.append(r * point_x)
        gesture_Y.append(r * point_y)

    centroid_x = (max(gesture_X) - min(gesture_X)) / 2
    centroid_y = (max(gesture_Y) - min(gesture_Y)) / 2
    scaled_X, scaled_Y = [], []
    for point_x, point_y in zip(gesture_X, gesture_Y):
        scaled_X.append(point_x - centroid_x)
        scaled_Y.append(point_y - centroid_y)
    return scaled_X, scaled_Y


def get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X,
                     valid_template_sample_points_Y):
    '''Get the shape score for every valid word after pruning.

    In this function, we should compare the sampled input gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a shape score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param valid_template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param valid_template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of shape scores.
    '''
    shape_scores = []
    if len(valid_template_sample_points_X) == 0 or len(valid_template_sample_points_Y) == 0:
        return shape_scores
    # Normalization is achieved by scaling the largest side of the bounding box to a
    # pre-determined  length L.
    L = 200
    # Scaled the template points, by calling the get_scaled_points function. The below loop takes (x,y) pairs which is required for our get_scaled_points function
    scaled_template_points_X, scaled_template_points_Y = [], []
    for template_points_X, template_points_Y in zip(valid_template_sample_points_X, valid_template_sample_points_Y):
        x_scaled, y_scaled = get_scaled_points(template_points_X, template_points_Y, L)
        scaled_template_points_X.append(x_scaled)
        scaled_template_points_Y.append(y_scaled)

    # Scaled the gesture points, by calling the get_scaled_points function
    gesture_sample_points_X, gesture_sample_points_Y = get_scaled_points(gesture_sample_points_X[0],
                                                                         gesture_sample_points_Y[0], L)
    # Finally, compute the Euclidean Norm and return the shape scores. Formula is given in paper: x = 1/N * sum(u-t)**2
    for template_points_X, template_points_Y in zip(scaled_template_points_X, scaled_template_points_Y):
        shape_scores.append(sum((math.sqrt((gesture_sample_points_X[j] - template_points_X[j]) ** 2 + (
                gesture_sample_points_Y[j] - template_points_Y[j]) ** 2)) for j in range(100)) / 100)

    return shape_scores


def get_small_d(p_X, p_Y, q_X, q_Y):
    min_distance = []
    for n in range(0, 100):
        distance = math.sqrt((p_X - q_X[n]) ** 2 + (p_Y - q_Y[n]) ** 2)
        min_distance.append(distance)
    return (sorted(min_distance)[0])


def get_big_d(p_X, p_Y, q_X, q_Y, r):
    final_max = 0
    for n in range(0, 100):
        local_max = 0
        distance = get_small_d(p_X[n], p_Y[n], q_X, q_Y)
        local_max = max(distance - r, 0)
        final_max += local_max
    return final_max


def get_delta(u_X, u_Y, t_X, t_Y, r, i):
    D1 = get_big_d(u_X, u_Y, t_X, t_Y, r)
    D2 = get_big_d(t_X, t_Y, u_X, u_Y, r)
    if D1 == 0 and D2 == 0:
        return 0
    else:
        return math.sqrt((u_X[i] - t_X[i]) ** 2 + (u_Y[i] - t_Y[i]) ** 2)


def get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X,
                        valid_template_sample_points_Y):
    '''Get the location score for every valid word after pruning.

    In this function, we should compare the sampled user gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a location score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of location scores.
    '''

    # Initialize location scores
    location_scores = np.zeros((len(valid_template_sample_points_X)))

    radius = 15

    alphas = np.zeros(100)

    for i in range(50):
        alphas[i] = (50 - i) / (51 * 50)

    for i in range(50, 100):
        alphas[i] = (i - 49) / (51 * 50)

    # Create a list of gesture points [[xi, yi]]
    gesture_points = [[gesture_sample_points_X[0][j], gesture_sample_points_Y[0][j]] for j in range(100)]

    # For each template
    for i in range(len(valid_template_sample_points_X)):
        # Create a list of template points
        template_points = [[valid_template_sample_points_X[i][j], valid_template_sample_points_Y[i][j]] for j in
                           range(100)]

        gesture_points = np.array(gesture_points).astype(np.float64)
        template_points = np.array(template_points).astype(np.float64)
        if np.any(np.isnan(gesture_points[0])):
            # Null point found in the gesture, so changing to gesture_points[0]
            gesture_points[0] = gesture_points[1]

        # Calculate distance of each gesture point with each template point. Calculating it manually was very slow, so used this
        distance_gesture_template = euclidean_distances(gesture_points, template_points)

        # Find distance of closest template point to each gesture point
        gesture_template_min_distances = np.min(distance_gesture_template, axis=1)

        # Find distance of closest gesture point to each template point
        template_gesture_min_distances = np.min(distance_gesture_template, axis=0)

        # If any gesture point is not within the radius tunnel or any template point is not within the radius tunnel
        if np.any(gesture_template_min_distances > radius) or np.any(template_gesture_min_distances > radius):
            # Calculate location score as sum of product of alpha and delta for each point. This formula is given in paper
            location_scores[i] = np.sum(np.multiply(alphas, np.diagonal(distance_gesture_template)))

    return location_scores


def get_integration_scores(shape_scores, location_scores):
    integration_scores = []
    # I am giving equal priority to both shape and location coefficient

    shape_coef = 0.5

    location_coef = 1 - shape_coef
    for i in range(len(shape_scores)):
        integration_scores.append(shape_coef * shape_scores[i] + location_coef * location_scores[i])
    return integration_scores


def get_best_words(valid_words, integration_scores):
    '''Get the best word.

    In this function, you should select top-n words with the highest integration scores and then use their corresponding
    probability (stored in variable "probabilities") as weight. The word with the highest weighted integration score is
    exactly the word we want.

    :param valid_words: A list of valid words.
    :param integration_scores: A list of corresponding integration scores of valid_words.
    :return: The most probable word list suggested to the user.
    '''

    n = 3

    # Find indices having the minimum score
    idxs = np.argsort(np.array(integration_scores))[:n]
    top_words_list = np.array(valid_words)[idxs]
    print("Top words in order are ", top_words_list)

    if len(top_words_list) == 0:
        return "Word not found"

    return top_words_list.tolist()


@app.route("/")
def init():
    return render_template('index.html')


@app.route('/shark2', methods=['POST'])
def shark2():
    start_time = time.time()
    data = json.loads(request.get_data())

    gesture_points_X = []
    gesture_points_Y = []
    for i in range(len(data)):
        gesture_points_X.append(data[i]['x'])
        gesture_points_Y.append(data[i]['y'])
    gesture_points_X = [gesture_points_X]
    gesture_points_Y = [gesture_points_Y]

    # All the points are given from browser mouse clicks
    # print(gesture_points_X)
    # print(gesture_points_Y)

    gesture_sample_points_X, gesture_sample_points_Y = generate_sample_points(gesture_points_X, gesture_points_Y)

    # We are sampling 100 sample points
    # print(gesture_sample_points_X)
    # print(gesture_sample_points_Y)

    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = do_pruning(gesture_points_X,
                                                                                             gesture_points_Y,
                                                                                             template_sample_points_X,
                                                                                             template_sample_points_Y)

    shape_scores = get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X,
                                    valid_template_sample_points_Y)

    location_scores = get_location_scores(gesture_sample_points_X, gesture_sample_points_Y,
                                          valid_template_sample_points_X, valid_template_sample_points_Y)

    integration_scores = get_integration_scores(shape_scores, location_scores)
    best_words = get_best_words(valid_words, integration_scores)

    end_time = time.time()
    integration_scores.sort()
    ret_str = dict()
    ret_str["elapsed_time"] = str(round((end_time - start_time) * 1000, 5)) + " ms"
    best_word_arr = []
    for itr in range(3):
        best_word_dict = dict()
        best_word_dict[best_words[itr]] = integration_scores[itr]
        best_word_arr.append(best_word_dict)
    ret_str["best_words"] = best_word_arr
    print(ret_str)
    return ret_str


if __name__ == "__main__":
    app.run()
