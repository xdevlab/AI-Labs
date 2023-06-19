import { RootState } from 'app/store/store';
import {
  ImageResizeInvocation,
  RandomIntInvocation,
  RangeOfSizeInvocation,
} from 'services/api';
import { NonNullableGraph } from 'features/nodes/types/types';
import { log } from 'app/logging/useLogger';
import {
  ITERATE,
  LATENTS_TO_IMAGE,
  MODEL_LOADER,
  NEGATIVE_CONDITIONING,
  NOISE,
  POSITIVE_CONDITIONING,
  RANDOM_INT,
  RANGE_OF_SIZE,
  IMAGE_TO_IMAGE_GRAPH,
  IMAGE_TO_LATENTS,
  LATENTS_TO_LATENTS,
  RESIZE,
} from './constants';
import { set } from 'lodash-es';
import { addControlNetToLinearGraph } from '../addControlNetToLinearGraph';

const moduleLog = log.child({ namespace: 'nodes' });

/**
 * Builds the Image to Image tab graph.
 */
export const buildLinearImageToImageGraph = (
  state: RootState
): NonNullableGraph => {
  const {
    positivePrompt,
    negativePrompt,
    model: model_name,
    cfgScale: cfg_scale,
    scheduler,
    steps,
    initialImage,
    img2imgStrength: strength,
    shouldFitToWidthHeight,
    width,
    height,
    iterations,
    seed,
    shouldRandomizeSeed,
  } = state.generation;

  /**
   * The easiest way to build linear graphs is to do it in the node editor, then copy and paste the
   * full graph here as a template. Then use the parameters from app state and set friendlier node
   * ids.
   *
   * The only thing we need extra logic for is handling randomized seed, control net, and for img2img,
   * the `fit` param. These are added to the graph at the end.
   */

  if (!initialImage) {
    moduleLog.error('No initial image found in state');
    throw new Error('No initial image found in state');
  }

  // copy-pasted graph from node editor, filled in with state values & friendly node ids
  const graph: NonNullableGraph = {
    id: IMAGE_TO_IMAGE_GRAPH,
    nodes: {
      [POSITIVE_CONDITIONING]: {
        type: 'compel',
        id: POSITIVE_CONDITIONING,
        prompt: positivePrompt,
      },
      [NEGATIVE_CONDITIONING]: {
        type: 'compel',
        id: NEGATIVE_CONDITIONING,
        prompt: negativePrompt,
      },
      [RANGE_OF_SIZE]: {
        type: 'range_of_size',
        id: RANGE_OF_SIZE,
        // seed - must be connected manually
        // start: 0,
        size: iterations,
        step: 1,
      },
      [NOISE]: {
        type: 'noise',
        id: NOISE,
      },
      [MODEL_LOADER]: {
        type: 'sd1_model_loader',
        id: MODEL_LOADER,
        model_name,
      },
      [LATENTS_TO_IMAGE]: {
        type: 'l2i',
        id: LATENTS_TO_IMAGE,
      },
      [ITERATE]: {
        type: 'iterate',
        id: ITERATE,
      },
      [LATENTS_TO_LATENTS]: {
        type: 'l2l',
        id: LATENTS_TO_LATENTS,
        cfg_scale,
        scheduler,
        steps,
        strength,
      },
      [IMAGE_TO_LATENTS]: {
        type: 'i2l',
        id: IMAGE_TO_LATENTS,
        // must be set manually later, bc `fit` parameter may require a resize node inserted
        // image: {
        //   image_name: initialImage.image_name,
        // },
      },
    },
    edges: [
      {
        source: {
          node_id: MODEL_LOADER,
          field: 'clip',
        },
        destination: {
          node_id: POSITIVE_CONDITIONING,
          field: 'clip',
        },
      },
      {
        source: {
          node_id: MODEL_LOADER,
          field: 'clip',
        },
        destination: {
          node_id: NEGATIVE_CONDITIONING,
          field: 'clip',
        },
      },
      {
        source: {
          node_id: MODEL_LOADER,
          field: 'vae',
        },
        destination: {
          node_id: LATENTS_TO_IMAGE,
          field: 'vae',
        },
      },
      {
        source: {
          node_id: RANGE_OF_SIZE,
          field: 'collection',
        },
        destination: {
          node_id: ITERATE,
          field: 'collection',
        },
      },
      {
        source: {
          node_id: ITERATE,
          field: 'item',
        },
        destination: {
          node_id: NOISE,
          field: 'seed',
        },
      },
      {
        source: {
          node_id: LATENTS_TO_LATENTS,
          field: 'latents',
        },
        destination: {
          node_id: LATENTS_TO_IMAGE,
          field: 'latents',
        },
      },
      {
        source: {
          node_id: IMAGE_TO_LATENTS,
          field: 'latents',
        },
        destination: {
          node_id: LATENTS_TO_LATENTS,
          field: 'latents',
        },
      },
      {
        source: {
          node_id: NOISE,
          field: 'noise',
        },
        destination: {
          node_id: LATENTS_TO_LATENTS,
          field: 'noise',
        },
      },
      {
        source: {
          node_id: MODEL_LOADER,
          field: 'vae',
        },
        destination: {
          node_id: IMAGE_TO_LATENTS,
          field: 'vae',
        },
      },
      {
        source: {
          node_id: MODEL_LOADER,
          field: 'unet',
        },
        destination: {
          node_id: LATENTS_TO_LATENTS,
          field: 'unet',
        },
      },
      {
        source: {
          node_id: NEGATIVE_CONDITIONING,
          field: 'conditioning',
        },
        destination: {
          node_id: LATENTS_TO_LATENTS,
          field: 'negative_conditioning',
        },
      },
      {
        source: {
          node_id: POSITIVE_CONDITIONING,
          field: 'conditioning',
        },
        destination: {
          node_id: LATENTS_TO_LATENTS,
          field: 'positive_conditioning',
        },
      },
    ],
  };

  // handle seed
  if (shouldRandomizeSeed) {
    // Random int node to generate the starting seed
    const randomIntNode: RandomIntInvocation = {
      id: RANDOM_INT,
      type: 'rand_int',
    };

    graph.nodes[RANDOM_INT] = randomIntNode;

    // Connect random int to the start of the range of size so the range starts on the random first seed
    graph.edges.push({
      source: { node_id: RANDOM_INT, field: 'a' },
      destination: { node_id: RANGE_OF_SIZE, field: 'start' },
    });
  } else {
    // User specified seed, so set the start of the range of size to the seed
    (graph.nodes[RANGE_OF_SIZE] as RangeOfSizeInvocation).start = seed;
  }

  // handle `fit`
  if (
    shouldFitToWidthHeight &&
    (initialImage.width !== width || initialImage.height !== height)
  ) {
    // The init image needs to be resized to the specified width and height before being passed to `IMAGE_TO_LATENTS`

    // Create a resize node, explicitly setting its image
    const resizeNode: ImageResizeInvocation = {
      id: RESIZE,
      type: 'img_resize',
      image: {
        image_name: initialImage.image_name,
      },
      is_intermediate: true,
      width,
      height,
    };

    graph.nodes[RESIZE] = resizeNode;

    // The `RESIZE` node then passes its image to `IMAGE_TO_LATENTS`
    graph.edges.push({
      source: { node_id: RESIZE, field: 'image' },
      destination: {
        node_id: IMAGE_TO_LATENTS,
        field: 'image',
      },
    });

    // The `RESIZE` node also passes its width and height to `NOISE`
    graph.edges.push({
      source: { node_id: RESIZE, field: 'width' },
      destination: {
        node_id: NOISE,
        field: 'width',
      },
    });

    graph.edges.push({
      source: { node_id: RESIZE, field: 'height' },
      destination: {
        node_id: NOISE,
        field: 'height',
      },
    });
  } else {
    // We are not resizing, so we need to set the image on the `IMAGE_TO_LATENTS` node explicitly
    set(graph.nodes[IMAGE_TO_LATENTS], 'image', {
      image_name: initialImage.image_name,
    });

    // Pass the image's dimensions to the `NOISE` node
    graph.edges.push({
      source: { node_id: IMAGE_TO_LATENTS, field: 'width' },
      destination: {
        node_id: NOISE,
        field: 'width',
      },
    });
    graph.edges.push({
      source: { node_id: IMAGE_TO_LATENTS, field: 'height' },
      destination: {
        node_id: NOISE,
        field: 'height',
      },
    });
  }

  // add controlnet
  addControlNetToLinearGraph(graph, LATENTS_TO_LATENTS, state);

  return graph;
};