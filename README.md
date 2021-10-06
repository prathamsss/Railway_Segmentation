# Railway_Segmentation

# Utils_&_Helper :
        - Contains scripts to

                   - Utils.py: Convert Dataset from Masks to COCO fromate (CMD Interface Not Avialable)
                   - Visualizer.py: Creates Visualisation for generated COCO formate (CMD Interface Not Avialable)
                   - Dataset_divide.py: Divides files into given number and saves in required dataset (CMD Avialable)
                   
                          
  # Detectron2 :
            For traning segmentation model, Pretrained model from Detectron2 Pytorch framwork was been imported.
            Mask RCNN along with FPN with RESNET 50 as backbone archetecture was been used.
            
            - Detectron2_Base: Notebook for traning model. Saves visualisation in /plots directory
            - model_2: Contains model file, and segmentation all segmentation outputs and results. 
                        
![alt text](https://github.com/prathamsss/Railway_Segmentation/blob/master/Detectron2/model_1/Output_imgs/Screenshot%202021-10-04%20at%2010.08.02%20PM.png)
![alt text](https://github.com/prathamsss/Railway_Segmentation/blob/master/Detectron2/model_1/Output_imgs/Screenshot%202021-10-04%20at%2010.08.02%20PM.png)
