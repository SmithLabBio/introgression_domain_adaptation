

center = -175 
rescaled = (center - 180) % 360
opposite = (rescaled + 180) % 360 - 180
mid1 = (rescaled + 90) % 360 - 180
mid2 = (rescaled + 270) % 360 - 180
print(mid1)
print(opposite)
print(mid2)