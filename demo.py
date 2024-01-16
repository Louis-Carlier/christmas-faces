import numpy

def demo(image : numpy.ndarray, antlers_guidance_scale = 4, santa_guidance_scale = 5, image_guidance_scale = 1.5) -> numpy.ndarray:
    """
    demo of the Christmas algorithm, author : L. Carlier, date : 1/24

    adds a santa hat to the furthest person detected and a red nose + antler reindeers to the closest person detected
    if there is only one person detected, add either the santa hat or the red nose + antler reindeers, with equal probability

    the framework consists of 3.5 steps :
    - step 1 : detect the faces on the picture using MTCNN -> arXiv link : https://arxiv.org/abs/1604.02878
    - step 2 : depth-segment the image using DPT -> arXiv link : https://arxiv.org/abs/2103.13413
    - step 2.5 : select the closest and furthest face by ordering the faces along their average depth
    - step 3 : generate the desired Christmas accessory using instructPix2Pix -> arXiv link : https://arxiv.org/abs/2211.09800
    
    inputs : 
    - image : input image, type : ndarray

    - antlers_guidance_scale : a coefficient that lets us control how much the image will be close to a reindeer (see intructPix2Pix docs), type : dfloat
    - antlers_guidance_scale : a coefficient that lets us control how much the image will be close to a santa (see intructPix2Pix docs), type : dfloat
    - antlers_guidance_scale : a coefficient that lets us control how much the image will be close to the original image (see intructPix2Pix docs), type : dfloat

    the last parameters are used to control the changed image depending if it looks to much like the original image or too different 

    outputs : 
    - changed image, type : ndarray
    """

    import numpy as np


    # STEP 1 : FACE DETECTION

    import mtcnn
    detector = mtcnn.MTCNN()
    faces = detector.detect_faces(image)
    n_faces  = len(faces)

    # ensure there is at least one face detected
    assert n_faces !=0, ('No face detected on image input.')


    # STEP 2 : DEPTH ESTIMATION

    from transformers import DPTImageProcessor, DPTForDepthEstimation
    import torch
    from PIL import Image

    image_pil = Image.fromarray(image)

    processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

    # prepare image for the model
    inputs = processor(images=image_pil, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image_pil.size[::-1],
        mode="bicubic",
        align_corners=False,
        )
    
    # depth estimation
    depth = prediction.squeeze().cpu().numpy()


    # SETP 2.5 : SELECT CLOSEST AND FURTHEST FACES 
    depth_list = []

    # compute mean estimated depth of each face
    for _ in range(len(faces)):
        y, x, h, w = faces[_]['box']
        mean_depth = np.mean(depth[x : x + w, y : y + h])
        depth_list.append(mean_depth)
    
    closest = np.argmax(depth_list)
    furthest = np.argmin(depth_list)

    same = False
    if closest == furthest:
        same = True

    
    # STEP 3 : SANTA HAT / RED NOSE + ANTLER REINDEER GENERATION

    y1, x1, h1, w1 = faces[closest]['box']
    y2, x2, h2, w2 = faces[furthest]['box']

    # dilatation coefficient to add some context to the faces
    alpha_x = 1.5
    alpha_y = 2.5

    x1, y1, w1, h1 = int(x1 + (1- alpha_x) * w1/2), int(y1 + (1- alpha_y) * h1/2), int(alpha_x * w1), int(alpha_y * h1)
    x2, y2, w2, h2 = int(x2 + (1- alpha_x) * w2/2), int(y2 + (1- alpha_y) * h2/2), int(alpha_x * w2), int(alpha_y * h2)

    closest_face = image[x1 : x1 + w1, y1 : y1 + h1, :]
    furthest_face = image[x2 : x2 + w2, y2 : y2 + h2, :]

    closest_face_pil = Image.fromarray(closest_face)
    furthest_face_pil = Image.fromarray(furthest_face)


    # create the new image we will modify
    new_image = image.copy()

    from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float32, safety_checker=None)
    pipe.to('cuda')
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    prompts = ['add red nose and reindeer antlers', 'wearing a santa hat']

    if same:
        coin = np.random.binomial(n=1, p=.5) # flips a coin

        new_face = pipe(prompts[coin], image=closest_face_pil,
                         num_inference_steps=50, guidance_scale = santa_guidance_scale if coin else antlers_guidance_scale, image_guidance_scale = image_guidance_scale
                         ).images[0]

        # PIL to ndarray
        new_face = np.array(new_face)

        # pix2pix crops a litle bit the images so we need to recover the new dimension
        nw, nh = new_face.shape[:-1]
        new_image[x1 : x1 + nw, y1 : y1 + nh, :] = new_face

    else:
        new_closest_face_pil = pipe(prompts[0], image=closest_face_pil,
                                     num_inference_steps=50, guidance_scale = santa_guidance_scale, image_guidance_scale = image_guidance_scale
                                     ).images[0]
        new_furthest_face_pil = pipe(prompts[1], image=furthest_face_pil,
                                      num_inference_steps=50, guidance_scale = antlers_guidance_scale, image_guidance_scale = image_guidance_scale
                                      ).images[0]

        # PIL to ndarray
        new_closest_face = np.array(new_closest_face_pil)
        new_furthest_face = np.array(new_furthest_face_pil)

        # pix2pix crops a litle bit the images so we need to recover the new dimension
        nw1, nh1 = new_closest_face.shape[:-1]
        nw2, nh2 = new_furthest_face.shape[:-1]

        new_image[x1 : x1 + nw1, y1 : y1 + nh1, :] = new_closest_face
        new_image[x2 : x2 + nw2, y2 : y2 + nh2, :] = new_furthest_face

    return(new_image)