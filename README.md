

# A Machine Learning Technique to Attenuate the Imprint of Chaotic Ocean Variability in Altimetric Observations

Mickaël Lalande, Redouane Lguensat, Sally Close and Thierry Penduff

[IGE - Institut des Géosciences de l'Environnement](http://www.ige-grenoble.fr/)<br/>



<br/>
This repository aims to make available the codezerr the preliminary study of the model choice section of the report. More details can be seen in the **notebooks** folder :
-  **visualize_data.ipynb** shows basics plots to visualize the data 
-  **plot_results.ipynb** shows some results of the filtering by the machine learning.


# Study zone 
We made our choice on a reduced zone (Agulhas eddies), in order to be able to test many different models. The choice of this zone was made to cover some chaotic zone (south of the map around eddies) and more forced area (north of the zone). We then applied these results to a bigger zone in the report.

![](zone.png)

# Example of result
The goal is to reconstruct the forced part of the total signal of sea level anomalies from 1979 to 1999 from an ensemble simulation with 50 members. The forced part is defined as the ensemble mean.

![](result.png)

#### This is only an additional material for the report. All details and further study can be found in there.
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE0NDQ3NDY4MDgsMTg0MDAyOTA4OSwtMj
A0NDE2MDIyNV19
-->