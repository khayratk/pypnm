try: paraview.simple
except: from paraview.simple import *
paraview.simple._DisableFirstRenderCameraReset()

Output00000 = GetActiveSource()
Glyph3 = Glyph( GlyphType="Arrow", GlyphTransform="Transform2" )

Glyph3.Scalars = ['POINTS', 'PoreRadius']
Glyph3.SetScaleFactor = 0.00019974855549999997
Glyph3.Vectors = ['POINTS', '']
Glyph3.GlyphTransform = "Transform2"
Glyph3.GlyphType = "Arrow"

Glyph3.SetScaleFactor = 1.0
Glyph3.RandomMode = 0
Glyph3.GlyphType = "Sphere"
Glyph3.ScaleMode = 'scalar'
Glyph3.MaskPoints = 0

Glyph3.GlyphType.Radius = 1.0

RenderView1 = GetRenderView()
Shrink2 = FindSource("Shrink2")
my_representation4 = GetDisplayProperties(Shrink2)
CellDatatoPointData2 = FindSource("CellDatatoPointData2")
my_representation5 = GetDisplayProperties(CellDatatoPointData2)
ExtractSurface2 = FindSource("ExtractSurface2")
my_representation6 = GetDisplayProperties(ExtractSurface2)
Tube2 = FindSource("Tube2")
my_representation7 = GetDisplayProperties(Tube2)
DataRepresentation6 = GetDisplayProperties(Output00000)
DataRepresentation7 = Show()
DataRepresentation7.EdgeColor = [0.0, 0.0, 0.5000076295109483]
DataRepresentation7.SelectionPointFieldDataArrayName = 'PoreRadius'
DataRepresentation7.SelectionCellFieldDataArrayName = 'TubeRadius'
DataRepresentation7.ColorArrayName = 'PoreRadius'
DataRepresentation7.ScaleFactor = 0.0002025323507950816

a1_PoreRadius_PVLookupTable = GetLookupTableForArray( "PoreRadius", 1 )

RenderView1.CameraClippingRange = [0.002311798038264228, 0.002696842621581536]

DataRepresentation7.LookupTable = a1_PoreRadius_PVLookupTable

Render()
