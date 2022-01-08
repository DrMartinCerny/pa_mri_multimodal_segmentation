import numpy as np
from sklearn.cluster import KMeans
from scipy import ndimage
import matplotlib.pyplot as plt

class KnospScore:

    def __init__(self, mask, fromArray=None):
        if fromArray is not None:
            # use only as object adapter for already known knosp score (ground truth or blackbox prediction)
            self.knosp_score_left = fromArray[0]
            self.knosp_score_right = fromArray[1]
            self.zurich_score, self.zurich_grade = None, None
        else:
            # determine knosp score from mask by geometrical method
            #mask = self.erode(mask)
            self.get_carotid_cross_sections(mask)
            self.get_dividing_lines(mask)
            self.knosp_score_left, self.knosp_score_right = self.get_knosp_score(mask)
            self.zurich_score, self.zurich_grade = self.get_zurich_score(mask)
    
    knospGrades = ['0', 'I', 'II', 'IIIa', 'IIIb', 'IV']
    zurichGrades = ['I', 'II', 'III', 'IV']
    
    def get_carotid_cross_sections(self, mask):

        points = []
        for o in range(128):
            for u in range(128):
                if mask[o,u] == 2:
                    points.append([o,u])

        kmeans = KMeans(n_clusters=4, random_state=0).fit(points).labels_

        clusters = [{'points': []} for u in range(4)]

        for i in range(len(clusters)):
            for o in range(len(points)):
                if kmeans[o] == i:
                    clusters[i]['points'].append(points[o])
        
        for cluster in clusters:
            cluster['points'] = np.array(cluster['points'])
            cluster['x'] = np.mean(cluster['points'][:,0])
            cluster['y'] = np.mean(cluster['points'][:,1])
            cluster['width'] = np.mean(np.abs(cluster['points'][:,0]-cluster['x']))*2
            cluster['height'] = np.mean(np.abs(cluster['points'][:,1]-cluster['y']))*2
            cluster['diameter'] = (cluster['width']+cluster['height'])/2

        clusters = list(sorted(clusters, key=lambda item: item['y']))
        left = list(sorted(clusters[:2], key=lambda item: item['x']))
        right = list(sorted(clusters[2:], key=lambda item: item['x']))
        left.extend(right)
        clusters = left

        self.left_supraclinoid = clusters[0]
        self.left_intracavernous = clusters[1]
        self.right_supraclinoid = clusters[2]
        self.right_intracavernous = clusters[3]

    def get_dividing_lines(self, mask):
        left_tangents = self.get_tangent(self.left_supraclinoid,self.left_intracavernous)
        right_tangents = self.get_tangent(self.right_supraclinoid,self.right_intracavernous)
        left_tangents = self.order_lines(left_tangents,(self.left_supraclinoid['x']+self.left_intracavernous['x'])/2)
        right_tangents = self.order_lines(right_tangents,(self.right_supraclinoid['x']+self.right_intracavernous['x'])/2)

        self.left_outter_tangent = left_tangents[0]
        self.left_midline = self.get_line(self.left_supraclinoid['x'], self.left_supraclinoid['y'], self.left_intracavernous['x'], self.left_intracavernous['y'])
        self.left_inner_tangent = left_tangents[1]
        self.left_perpendicular = self.get_kolmou(left_tangents[0], (self.left_intracavernous['x'],self.left_intracavernous['y']))
        self.left_upper_limit = self.get_kolmou(left_tangents[0], (self.left_supraclinoid['x'],self.left_supraclinoid['y']))
        self.right_inner_tangent = right_tangents[0]
        self.right_midline = self.get_line(self.right_supraclinoid['x'], self.right_supraclinoid['y'], self.right_intracavernous['x'], self.right_intracavernous['y'])
        self.right_outter_tangent = right_tangents[1]
        self.right_perpendicular = self.get_kolmou(right_tangents[1], (self.right_intracavernous['x'],self.right_intracavernous['y']))
        self.right_upper_limit = self.get_kolmou(right_tangents[0], (self.right_supraclinoid['x'],self.right_supraclinoid['y']))

    def get_subspace(self, space,left_border,right_border):
        subspace = np.zeros(space.shape)
        for i1 in range (space.shape[0]):
            for i2 in range (space.shape[0]):
                if left_border is None or left_border['a'] * i1 - i2 + left_border['b'] < 0:
                    if right_border is None or right_border['a'] * i1 - i2 + right_border['b'] > 0:
                        subspace[i1,i2] = space[i1,i2]
        return subspace
    
    def get_knosp_score(self, prediction):
        prediction = prediction==1
        left = 0
        right = 0

        left_subspace = self.get_subspace(prediction,None,self.left_upper_limit)
        if np.sum(self.get_subspace(left_subspace,None,self.left_outter_tangent)) > 0:
            subspace = self.get_subspace(left_subspace,None,self.left_outter_tangent)
            if np.sum(self.get_subspace(subspace,self.left_perpendicular,None)) > 0:
                if np.sum(self.get_subspace(subspace,None,self.left_perpendicular)) > 0:
                    left = 5
                else: left = 3
            else: left = 4
        elif np.sum(self.get_subspace(left_subspace,self.left_outter_tangent,self.left_midline)) > 0:
            left = 2
        elif np.sum(self.get_subspace(left_subspace,self.left_midline,self.left_inner_tangent)) > 0:
            left = 1
        
        right_subspace = self.get_subspace(prediction,None,self.right_upper_limit)
        if np.sum(self.get_subspace(right_subspace,self.right_outter_tangent,None)) > 0:
            subspace = self.get_subspace(right_subspace,self.right_outter_tangent,None)
            if np.sum(self.get_subspace(subspace,self.right_perpendicular,None)) > 0:
                if np.sum(self.get_subspace(subspace,None,self.right_perpendicular)) > 0:
                    right = 5
                else: right = 3
            else: right = 4
        elif np.sum(self.get_subspace(right_subspace,self.right_midline,self.right_outter_tangent)) > 0:
            right = 2
        elif np.sum(self.get_subspace(right_subspace,self.right_inner_tangent,self.right_midline)) > 0:
            right = 1
        return left, right
    
    def get_zurich_score(self, prediction):
        prediction = prediction==1
        self.zps_max_diameter_y = np.argmax(np.sum(prediction, axis=-1))
        self.zps_max_diameter_x_start = np.argmax(prediction, axis=-1)*(np.sum(prediction, axis=-1)>0)
        self.zps_max_diameter_x_end = 128-np.argmax(np.flip(prediction, axis=-1), axis=-1)*(np.sum(prediction, axis=-1)>0)
        self.zps_max_diameter_x_end[np.sum(prediction, axis=-1)==0] = 0
        self.zps_max_diameter = self.zps_max_diameter_x_end-self.zps_max_diameter_x_start
        self.zps_max_diameter_y = np.argmax(self.zps_max_diameter)
        self.zps_max_diameter = self.zps_max_diameter[self.zps_max_diameter_y]
        self.zps_intercarotid_distance = np.sqrt(np.power(self.left_intracavernous['x']-self.right_intracavernous['x'],2)+np.power(self.left_intracavernous['y']-self.right_intracavernous['y'],2))
        zps = self.zps_max_diameter / self.zps_intercarotid_distance
        self.zps_max_diameter_x_start = self.zps_max_diameter_x_start[self.zps_max_diameter_y]
        self.zps_max_diameter_x_end = self.zps_max_diameter_x_end[self.zps_max_diameter_y]
        has_encasement = self.knosp_score_left == 5 or self.knosp_score_right == 5
        return zps, 3 if has_encasement else 0 if zps < 0.75 else 1 if zps < 1.25 else 2

    def erode(self, maska):
        eroded = np.zeros(maska.shape,dtype=np.int16)
        for i in range(1,np.max(maska)+1):
            err = ndimage.binary_erosion(maska==i,iterations=2)
            err = ndimage.binary_dilation(err,iterations=2)
            eroded[err] = i
        return eroded
    
    def get_line(self, x1,y1,x2,y2):
        a = (y2-y1)/(x2-x1)
        b = y1 - a*x1
        alfa = np.arccos
        return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'a': a, 'b': b}

    def line_start(self, line, UPSAMPLING):
        return (int(line['b']*UPSAMPLING),0)

    def line_end(self, line, UPSAMPLING, canvas_size):
        return (int((line['b']+canvas_size*line['a'])*UPSAMPLING),int(canvas_size*UPSAMPLING))

    def line_to_point_distance(self, line, x, y):
        return np.abs((line['b']-y)-((-x)*line['a']))/np.sqrt(np.power(line['a'],2)+1)

    def circle_intersection(self, x0, y0, r0, x1, y1, r1):

        d = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        if d > r0 + r1:             # non intersecting
            return None
        if d < abs(r0 - r1):        # one circle within other
            return None
        if d == 0 and r0 == r1:     # coincident circles
            return None

        a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
        h = np.sqrt(r0 ** 2 - a ** 2)
        x2 = x0 + a * (x1 - x0) / d
        y2 = y0 + a * (y1 - y0) / d
        x3 = x2 + h * (y1 - y0) / d
        y3 = y2 - h * (x1 - x0) / d

        x4 = x2 - h * (y1 - y0) / d
        y4 = y2 + h * (x1 - x0) / d

        return (x3, y3), (x4, y4)
        
    def pick_external_tangent(self, line,shift,x,y,d):
        tangent1 = {'a': line['a'], 'b': line['b']+shift}
        tangent2 = {'a': line['a'], 'b': line['b']-shift}
        #print(d, line_to_point_distance(tangent1,x,y))
        #print(d, line_to_point_distance(tangent2,x,y))
        if (np.abs(self.line_to_point_distance(tangent1,x,y)-d) < np.abs(self.line_to_point_distance(tangent2,x,y)-d)):
            return tangent1
        else:
            return tangent2

    def get_tangent(self, cluster1,cluster2,image=None):
        if cluster1['diameter'] < cluster2['diameter']:
            cc = cluster1
            cluster1 = cluster2
            cluster2 = cc
        x1 = cluster1['x']
        y1 = cluster1['y']
        x2 = cluster2['x']
        y2 = cluster2['y']
        d1 = cluster1['diameter']
        d2 = cluster2['diameter']
        xh = (x1+x2)/2
        yh = (y1+y2)/2
        d_half = np.sqrt(np.power(x1-x2, 2)+np.power(y1-y2, 2))/2
        d_dif = d1-d2
        tangents = []
        for inter in self.circle_intersection(xh, yh, d_half, x1, y1, d_dif):
            tangent = self.get_line(inter[0],inter[1],x2,y2)
            shift = np.sqrt(d2*d2 - tangent['a'] * tangent['a'] + 1)
            tangent = self.pick_external_tangent(tangent,shift,x1,y1,d1)
            tangents.append(tangent)
        return tangents

    def get_kolmou(self, line,pass_point):
        alfa = np.arctan(1/line['a'])
        a = np.tan(alfa)
        a = 999999
        b = pass_point[1] - a * pass_point[0]
        kolma = {'a': a, 'b': b}
        return kolma

    def order_lines(self, lines, mindpointY):
        line1x = mindpointY * lines[0]['a'] + lines[0]['b']
        line2x = mindpointY * lines[1]['a'] + lines[1]['b']
        #print(mindpointY, line1x, line2x)
        if line1x < line2x:
            return lines
        else:
            return[lines[1],lines[0]]