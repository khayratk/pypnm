try: paraview.simple
except: from paraview.simple import *
paraview.simple._DisableFirstRenderCameraReset()

Output000000_vtk = GetActiveSource()
Shrink1 = Shrink()

Shrink1.ShrinkFactor = 1.0

RenderView1 = GetRenderView()
DataRepresentation1 = GetDisplayProperties(Output000000_vtk)
Glyph1 = FindSource("Glyph1")
DataRepresentation2 = GetDisplayProperties(Glyph1)
DataRepresentation3 = Show()
DataRepresentation3.EdgeColor = [0.0, 0.0, 0.5000076295109483]
DataRepresentation3.SelectionPointFieldDataArrayName = 'PoreRadius'
DataRepresentation3.SelectionCellFieldDataArrayName = 'TubeRadius'
DataRepresentation3.ColorArrayName = 'PoreRadius'
DataRepresentation3.ScalarOpacityUnitDistance = 0.00015209299753087918
DataRepresentation3.ScaleFactor = 0.0001997485516994857

CellDatatoPointData1 = CellDatatoPointData()

a1_PoreRadius_PVLookupTable = GetLookupTableForArray( "PoreRadius", 1 )

DataRepresentation1.Visibility = 0

DataRepresentation3.ScalarOpacityFunction = []
DataRepresentation3.LookupTable = a1_PoreRadius_PVLookupTable

DataRepresentation4 = Show()
DataRepresentation4.EdgeColor = [0.0, 0.0, 0.5000076295109483]
DataRepresentation4.SelectionPointFieldDataArrayName = 'TubeRadius'
DataRepresentation4.SelectionCellFieldDataArrayName = 'TubeRadius'
DataRepresentation4.ColorArrayName = 'PoreRadius'
DataRepresentation4.ScalarOpacityUnitDistance = 0.00015209299753087918
DataRepresentation4.ScaleFactor = 0.0001997485516994857

ExtractSurface1 = ExtractSurface()

a1_TubeRadius_PVLookupTable = GetLookupTableForArray( "TubeRadius", 1, NanColor=[0.25, 0.0, 0.0], RGBPoints=[3.192349e-06, 0.23, 0.299, 0.754, 9.524875e-06, 0.706, 0.016, 0.15], VectorMode='Magnitude', ColorSpace='Diverging', ScalarRangeInitialized=1.0 )

a1_TubeRadius_PiecewiseFunction = CreatePiecewiseFunction( Points=[0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0] )

DataRepresentation3.Visibility = 0

DataRepresentation4.ScalarOpacityFunction = a1_TubeRadius_PiecewiseFunction
DataRepresentation4.ColorArrayName = 'TubeRadius'
DataRepresentation4.LookupTable = a1_TubeRadius_PVLookupTable

DataRepresentation5 = Show()
DataRepresentation5.EdgeColor = [0.0, 0.0, 0.5000076295109483]
DataRepresentation5.SelectionPointFieldDataArrayName = 'TubeRadius'
DataRepresentation5.SelectionCellFieldDataArrayName = 'TubeRadius'
DataRepresentation5.ColorArrayName = 'TubeRadius'
DataRepresentation5.ScaleFactor = 0.0001997485516994857

Tube1 = Tube()

DataRepresentation4.Visibility = 0

DataRepresentation5.LookupTable = a1_TubeRadius_PVLookupTable

Tube1.Scalars = ['POINTS', 'TubeRadius']
Tube1.Vectors = ['POINTS', '']
Tube1.Radius = 1.997485516994857e-05

Tube1.VaryRadius = 'By Absolute Scalar'
Tube1.Radius = 1.0
Tube1.RadiusFactor = 20.0

DataRepresentation6 = Show()
DataRepresentation6.EdgeColor = [0.0, 0.0, 0.5000076295109483]
DataRepresentation6.SelectionPointFieldDataArrayName = 'TubeRadius'
DataRepresentation6.SelectionCellFieldDataArrayName = 'TubeRadius'
DataRepresentation6.ColorArrayName = 'TubeRadius'
DataRepresentation6.ScaleFactor = 0.00020080167341802736

RenderView1.CameraPosition = [0.0008967092009377338, 0.0003316853849846356, 0.003625105488695014]
RenderView1.CameraClippingRange = [0.0035491079063353203, 0.0037295147601305767]

DataRepresentation5.Visibility = 0

DataRepresentation6.LookupTable = a1_TubeRadius_PVLookupTable

Render()
