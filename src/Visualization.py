import numpy as np
import cv2

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
    
    def drawKnospLines(img,knospScore,upsampling=2):
        img = img.copy()
        img = Visualization.upsample(img,upsampling=upsampling)

        for cluster in [knospScore.left_supraclinoid,knospScore.right_supraclinoid,knospScore.left_intracavernous,knospScore.right_intracavernous]:
            #kolecka kolem karotid
            cv2.circle(img, (int(cluster['y']*upsampling), int(cluster['x']*upsampling)), 2, (0, 0, 255), 2)
            cv2.circle(img, (int(cluster['y']*upsampling), int(cluster['x']*upsampling)), int(cluster['diameter']*upsampling), (0, 0, 255), 1)
        

        #lines
        cv2.line(img,knospScore.line_start(knospScore.left_outter_tangent,upsampling),knospScore.line_end(knospScore.left_outter_tangent,upsampling,128),(0, 0, 255),1)
        cv2.line(img,knospScore.line_start(knospScore.left_midline,upsampling),knospScore.line_end(knospScore.left_midline,upsampling,128),(0, 0, 255),1)
        cv2.line(img,knospScore.line_start(knospScore.left_inner_tangent,upsampling),knospScore.line_end(knospScore.left_inner_tangent,upsampling,128),(0, 0, 255),1)
        #cv2.line(img,knospScore.line_start(knospScore.left_perpendicular,upsampling),knospScore.line_end(knospScore.left_perpendicular,upsampling,128),(0, 0, 255),1)
        #cv2.line(img,knospScore.line_start(knospScore.left_upper_limit,upsampling),knospScore.line_end(knospScore.left_upper_limit,upsampling,128),(0, 0, 255),1)
        cv2.line(img,knospScore.line_start(knospScore.right_inner_tangent,upsampling),knospScore.line_end(knospScore.right_inner_tangent,upsampling,128),(0, 0, 255),1)
        cv2.line(img,knospScore.line_start(knospScore.right_midline,upsampling),knospScore.line_end(knospScore.right_midline,upsampling,128),(0, 0, 255),1)
        cv2.line(img,knospScore.line_start(knospScore.right_outter_tangent,upsampling),knospScore.line_end(knospScore.right_outter_tangent,upsampling,128),(0, 0, 255),1)
        #cv2.line(img,knospScore.line_start(knospScore.right_perpendicular,upsampling),knospScore.line_end(knospScore.right_perpendicular,upsampling,128),(0, 0, 255),1)
        #cv2.line(img,knospScore.line_start(knospScore.right_upper_limit,upsampling),knospScore.line_end(knospScore.right_upper_limit,upsampling,128),(0, 0, 255),1)
        
        knosp_names = ['0', 'I', 'II', 'IIIa', 'IIIb', 'IV']
        cv2.putText(img, "Geo:      Left: Knosp "+knosp_names[knospScore.knosp_score_left]+" / Right: Knosp "+knosp_names[knospScore.knosp_score_right], (10,15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255))
        #cv2.putText(img, "Blackbox: Left: Knosp "+knosp_names[knospScore.knosp_score_left]+" / Right: Knosp "+knosp_names[knospScore.knosp_score_right], (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255))
        #cv2.putText(img, "GT:       Left: Knosp "+knosp_names[knospScore.knosp_score_left]+" / Right: Knosp "+knosp_names[knospScore.knosp_score_right], (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255))
        
        return img