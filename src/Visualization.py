import numpy as np
import cv2

from src.KnospScore import KnospScore

class Visualization:

    def toBitmap(img,intensity_range=2):
        # transforms image from dataset to RGB bitmap
        img = np.clip(img,-intensity_range,intensity_range)
        img = ((img+intensity_range)/(intensity_range*2))
        return np.stack([img,img,img],axis=-1)
    
    def overlay(img,mask,a=0.7):
        # overlay with tumor and ICA labels
        img = img.copy()
        img[mask==1,0] *= a
        img[mask==1,2] *= a
        img[mask==2,0] *= a
        img[mask==2,1] *= a
        return img
    
    def upsample(img,upsampling=2):
        for i in range(upsampling-1):
            img = cv2.pyrUp(img)
        return img
    
    def drawKnospLines(img,knospScoreGroundTruth,knospScoreGeometric,knospScoreBlackbox=None,upsampling=2):
        img = img.copy()
        img = Visualization.upsample(img,upsampling=upsampling)

        for cluster in [knospScoreGeometric.left_supraclinoid,knospScoreGeometric.right_supraclinoid,knospScoreGeometric.left_intracavernous,knospScoreGeometric.right_intracavernous]:
            #kolecka kolem karotid
            cv2.circle(img, (int(cluster['y']*upsampling), int(cluster['x']*upsampling)), 2, (0, 0, 255), 2)
            cv2.circle(img, (int(cluster['y']*upsampling), int(cluster['x']*upsampling)), int(cluster['diameter']*upsampling), (0, 0, 255), 1)
        

        #lines
        try: cv2.line(img,knospScoreGeometric.line_start(knospScoreGeometric.left_outter_tangent,upsampling),knospScoreGeometric.line_end(knospScoreGeometric.left_outter_tangent,upsampling,128),(0, 0, 255),1)
        except: pass
        try: cv2.line(img,knospScoreGeometric.line_start(knospScoreGeometric.left_midline,upsampling),knospScoreGeometric.line_end(knospScoreGeometric.left_midline,upsampling,128),(0, 0, 255),1)
        except: pass
        try: cv2.line(img,knospScoreGeometric.line_start(knospScoreGeometric.left_inner_tangent,upsampling),knospScoreGeometric.line_end(knospScoreGeometric.left_inner_tangent,upsampling,128),(0, 0, 255),1)
        except: pass
        #try: cv2.line(img,knospScore.line_start(knospScore.left_perpendicular,upsampling),knospScore.line_end(knospScore.left_perpendicular,upsampling,128),(0, 0, 255),1)
        #except: pass
        #try: cv2.line(img,knospScore.line_start(knospScore.left_upper_limit,upsampling),knospScore.line_end(knospScore.left_upper_limit,upsampling,128),(0, 0, 255),1)
        #except: pass
        try: cv2.line(img,knospScoreGeometric.line_start(knospScoreGeometric.right_inner_tangent,upsampling),knospScoreGeometric.line_end(knospScoreGeometric.right_inner_tangent,upsampling,128),(0, 0, 255),1)
        except: pass
        try: cv2.line(img,knospScoreGeometric.line_start(knospScoreGeometric.right_midline,upsampling),knospScoreGeometric.line_end(knospScoreGeometric.right_midline,upsampling,128),(0, 0, 255),1)
        except: pass
        try: cv2.line(img,knospScoreGeometric.line_start(knospScoreGeometric.right_outter_tangent,upsampling),knospScoreGeometric.line_end(knospScoreGeometric.right_outter_tangent,upsampling,128),(0, 0, 255),1)
        except: pass
        #try: cv2.line(img,knospScore.line_start(knospScore.right_perpendicular,upsampling),knospScore.line_end(knospScore.right_perpendicular,upsampling,128),(0, 0, 255),1)
        #except: pass
        #try: cv2.line(img,knospScore.line_start(knospScore.right_upper_limit,upsampling),knospScore.line_end(knospScore.right_upper_limit,upsampling,128),(0, 0, 255),1)
        #except: pass
        
        Visualization.addKnospScoreLabels(img,knospScoreGroundTruth,'GT:       ',15)
        Visualization.addKnospScoreLabels(img,knospScoreGeometric,'Geo:      ',30)
        if knospScoreBlackbox is not None:
            Visualization.addKnospScoreLabels(img,knospScoreBlackbox,'Blackbox: ',45)

        return img
    
    def addKnospScoreLabels(img,knospScore,method,y):
        knosp_score_left = KnospScore.knospGrades[knospScore.knosp_score_left] if knospScore.knosp_score_left is not None else '??'
        knosp_score_right = KnospScore.knospGrades[knospScore.knosp_score_right] if knospScore.knosp_score_right is not None else '??'
        cv2.putText(img, method+"Left: Knosp "+knosp_score_left+" / Right: Knosp "+knosp_score_right, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255))