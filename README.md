# Master-Thesis

In this repository the code for all the experiments and analysis performed for my Master's thesis is stored. The thesis was done at the University of Groningen in the study year 2020-2021. 

Abstract:
In this thesis, the multi-domain tasks of visual question answering (VQA) and image generation (IG) are explored. 
VQA is the task of answering open-ended questions about an image. IG is the task of synthesizing a semantically matching high-quality image from a text description. Models for these tasks are not showing a deep understanding of their tasks. To explore their abilities their generalizability and robustness are explored. To relate these two tasks more the IG task was modified to synthesize from a question-answer (QA) pair. By making this modification, the models can be employed in a cyclical architecture. The applications of cyclical architectures towards improving robustness and generalizability were explored. Also, the effectiveness of a semantic consistency metric using a VQA model as a critic towards the IG images is explored. The study was performed on data set with images of abstract shapes to improve prototyping speed and analysis possibility. The data set was designed to require a lot of generalizability. The results showed that VQA and IG models are poor at spatial understanding. The VQA models were able to show degrees of generalizability, but the IG models displayed look-up-table-like behavior often failing to capture the essence of the QA pairs lacking generalizability. The VQA consistency metric proved to be an effective way to measure semantic consistency. Lastly, the VQA model can learn more variation through a cyclical architecture with the IG model.

PDF:
https://fse.studenttheses.ub.rug.nl/25658/
