<h1>Philly Vehicle Locator</h1>
<h2>Background</h2>
<h4>What is the Philly Vehicle Locator (PVL)?</h4>
Philly Vehicle Locator (PVL) is the City of Philadelphia's custom vehicle tracking solution. The PVL system 
has been designed to make GPS data received from departmental vehicles operational 
and is managed by the Office of Innovation & Technology's CityGeo team.
<h4>How is PVL different from AVL?</h4>
PVL has been designed to make GPS data easier to analyze, more readily available, and more efficient to store and 
process.
<h5>Efficiency & Ease of Analysis</h5>
By aggregating vehicle GPS points directly to street centerline segments, PVL is able to remove redundancies found in 
raw GPS feeds.  This reduces record storage by 75% and makes querying & filtering for mapping and reporting much 
quicker.
<h5>Data Availability</h5>
Raw GPS points and PVL processed vehicle route data are made internally available to City employees via shared database servers.  
<h2>Requirements, Configuration & Environment</h2>
<h4>Requirements</h4>
Philly Vehicle Locator (base configuration) requires the following:
<ul>
<li>Python 3.x
<ul>
<li>arcpy</li>
<li>psycopg2</li>
<li>networkx</li>
</ul>
</li>
<li>PostgreSQL Database
<ul>
<li>Grid Index</li>
<li>Street Centerline Data</li>
</ul>
</li>
</ul>
<b>NOTE -</b> Frequency of GPS points consumed by the philly-vehicle-locator.py script will heavily influence successful 
matching of vehicle GPS points to street centerline segments.  The City of Philadelphia is currently utilizing vehicle 
GPS points received at a 15 second ping rate.
<h4>Environment</h4>
The Office of Innovation & Technology is currently using the following technologies & software for PVL:
<ul>
<li>Amazon Web Services
<ul><li>EC2</li>
<li>RDS</li>
</ul>
</li>
<li>Esri ArcGIS GeoEvent Server 10.6</li>
<li>Esri ArcGIS Server 10.6/10.7</li>
<li>PostgreSQL 10.6</li>
<li>Python 3.6</li>
<li>Verizon NetworkFleet</li>
</ul>
<h2>Acknowledgments & Licensing</h2>
<h5>Acknowledgements</h5>
The original basis of the Philly Vehicle Locator scripts were derived from <b>
<a href="https://github.com/simonscheider">simonscheider's</a></b> repository, <b>
<a href="https://github.com/simonscheider/mapmatching">mapmatching</a></b>.
<h5>Licensing</h5>
[<i>Coming Soon</i>]
  
