import Metashape
import os, sys, time
import argparse


def find_files(folder: str, types: str):
    return [entry.path for entry in os.scandir(folder) if
            (entry.is_file() and os.path.splitext(entry.name)[1].lower() in types)]

def apply_mask_to_all_images(chunk, mask_path):
    for camera in chunk.cameras:
        if not camera.mask:  # 如果当前没有掩码
            camera.mask = Metashape.Mask()  
            print('mask is loaded')
            camera.mask.load(mask_path)
def general_worlflow(image_folder: str, output_folder: str,
                     mask_file: str,
                     texture_size: int,
                     keypoint_limit: int,
                     tiepoint_limit: int):
    # Checking compatibility
    compatible_major_version = "2.1"
    found_major_version = ".".join(Metashape.app.version.split('.')[:2])
    if found_major_version != compatible_major_version:
        raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version,
                                                                          compatible_major_version))

    photos = find_files(image_folder, [".jpg", ".jpeg", ".png", ".tif", ".tiff"])

    doc = Metashape.Document()
    doc.save(os.path.join(output_folder, 'project.psx'))

    chunk = doc.addChunk()

    chunk.addPhotos(photos)
    if mask_file:
        chunk.generateMasks(path=mask_file, masking_mode=Metashape.MaskingMode.MaskingModeFile, 
                            cameras=chunk.cameras)


    print(str(len(chunk.cameras)) + " images loaded")

    chunk.matchPhotos(downscale=0,downscale_3d=1,
                      keypoint_limit=keypoint_limit, tiepoint_limit=tiepoint_limit,
                      generic_preselection=True, reference_preselection=False,filter_mask=False)
   
    doc.save()

    chunk.alignCameras()
    doc.save()

    chunk.buildDepthMaps(downscale=1, filter_mode=Metashape.MildFiltering)
    doc.save()
    
    chunk.buildPointCloud()
    doc.save()

    chunk.buildModel(source_data=Metashape.DepthMapsData, face_count=Metashape.HighFaceCount)
    doc.save()

    chunk.buildUV(page_count=1, texture_size=texture_size)
    doc.save()

    chunk.buildTexture(texture_size=texture_size, ghosting_filter=True)
    # doc.save()

    has_transform = chunk.transform.scale and chunk.transform.rotation and chunk.transform.translation

    if has_transform:
        chunk.buildPointCloud()
        doc.save()

        chunk.buildDem(source_data=Metashape.PointCloudData)
        doc.save()

        chunk.buildOrthomosaic(surface_data=Metashape.ElevationData)
        doc.save()

    if chunk.model:
        chunk.exportModel(output_folder + '/model.obj')

    if chunk.point_cloud:
        chunk.exportPointCloud(output_folder + '/point_cloud.ply', source_data = Metashape.PointCloudData)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Metashape general workflow')
    # Required arguments
    parser.add_argument('--image_folder', help='Input image folder.')
    parser.add_argument('--output_folder', default= './',help='Output folder.')
    # Optional arguments
    parser.add_argument('--texture_size', default=4096, type=int,
                        choices=[1024, 2048, 4096, 8192, 16384],
                        help='Texture size')
    parser.add_argument('--keypoint_limit', default=40000, type=int,
                        help='matchPhotos keypoint limit')
    parser.add_argument('--tiepoint_limit', default=10000, type=int,
                        help='matchPhotos tiepoint limit')

    parser.add_argument('--mask_file', default=None, help='Mask file.')

    args = parser.parse_args()

    general_worlflow(image_folder=args.image_folder,
                     output_folder=args.image_folder,
                     mask_file=args.mask_file,
                     texture_size=args.texture_size,
                     keypoint_limit=args.keypoint_limit,
                     tiepoint_limit=args.tiepoint_limit)
