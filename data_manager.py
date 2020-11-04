import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("bigquery-results.csv", sep=";")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Oslo lat long area
oslo_lat_lon = (10.7027, 10.7772, 59.9112, 59.9438)
df = df[
    (oslo_lat_lon[0] <= df["lon"])
    & (df["lon"] <= oslo_lat_lon[1])
    & (oslo_lat_lon[2] <= df["lat"])
    & (df["lat"] <= oslo_lat_lon[3])
].head(10)
print(len(df))
print(df[["lat", "lon"]].head())

fig, ax = plt.subplots()
ax.scatter(df["lon"], df["lat"], zorder=1, alpha=0.5, s=10)
ax.set_xlim(oslo_lat_lon[0], oslo_lat_lon[1])
ax.set_ylim(oslo_lat_lon[2], oslo_lat_lon[3])

oslo = plt.imread("oslo.png")
ax.imshow(oslo, zorder=0, extent=oslo_lat_lon, aspect="equal", alpha=0.4)
plt.show()
