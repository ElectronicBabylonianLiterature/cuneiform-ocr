from typing import Tuple, Optional, Union, Sequence, List, Any

import cv2
import numpy as np
import pyclipper
from shapely import Polygon, MultiPolygon

TextDetDataSample = Any
ArrayLike = Any

def offset_polygon(poly: ArrayLike, distance: float) -> ArrayLike:
    """Offset (expand/shrink) the polygon by the target distance. It's a
    wrapper around pyclipper based on Vatti clipping algorithm.

    Warning:
        Polygon coordinates will be casted to int type in PyClipper. Mind the
        potential precision loss caused by the casting.

    Args:
        poly (ArrayLike): A polygon. In any form can be converted
            to an 1-D numpy array. E.g. list[float], np.ndarray,
            or torch.Tensor. Polygon is written in
            [x1, y1, x2, y2, ...].
        distance (float): The offset distance. Positive value means expanding,
            negative value means shrinking.

    Returns:
        np.array: 1-D Offsetted polygon ndarray in float32 type. If the
        result polygon is invalid or has been split into several parts,
        return an empty array.
    """
    poly = np.array(poly).reshape(-1, 2)
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    # Returned result will be in type of int32, convert it back to float32
    # following MMOCR's convention
    result = np.array(pco.Execute(distance), dtype=object)
    if len(result) > 0 and isinstance(result[0], list):
        # The processed polygon has been split into several parts
        result = np.array([])
    result = result.astype(np.float32)
    # Always use the first polygon since only one polygon is expected
    # But when the resulting polygon is invalid, return the empty array
    # as it is
    return result if len(result) == 0 else result[0].flatten()

def poly2shapely(polygon) -> Polygon:
    """Convert a polygon to shapely.geometry.Polygon.

    Args:
        polygon (ArrayLike): A set of points of 2k shape.

    Returns:
        polygon (Polygon): A polygon object.
    """
    polygon = np.array(polygon, dtype=np.float32)
    assert polygon.size % 2 == 0 and polygon.size >= 6

    polygon = polygon.reshape([-1, 2])
    return Polygon(polygon)

def polys2shapely(polygons):
    """Convert a nested list of boundaries to a list of Polygons.

    Args:
        polygons (list): The point coordinates of the instance boundary.

    Returns:
        list: Converted shapely.Polygon.
    """
    return [poly2shapely(polygon) for polygon in polygons]

def fill_hole(input_mask) -> np.array:
    """Fill holes in matrix.

        Input:
            [[0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 1, 0],
             [0, 1, 0, 0, 0, 1, 0],
             [0, 1, 1, 1, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0]]
        Output:
            [[0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 1, 0],
             [0, 1, 1, 1, 1, 1, 0],
             [0, 1, 1, 1, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0]]

    Args:
        input_mask (ArrayLike): The input mask.

    Returns:
        np.array: The output mask that has been filled.
    """
    input_mask = np.array(input_mask)
    h, w = input_mask.shape
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    canvas[1:h + 1, 1:w + 1] = input_mask.copy()

    mask = np.zeros((h + 4, w + 4), np.uint8)

    cv2.floodFill(canvas, mask, (0, 0), 1)
    canvas = canvas[1:h + 1, 1:w + 1].astype(np.bool_)

    return ~canvas | input_mask



def poly_make_valid(poly: Polygon) -> Polygon:
    """Convert a potentially invalid polygon to a valid one by eliminating
    self-crossing or self-touching parts. Note that if the input is a line, the
    returned polygon could be an empty one.

    Args:
        poly (Polygon): A polygon needed to be converted.

    Returns:
        Polygon: A valid polygon, which might be empty.
    """
    assert isinstance(poly, Polygon)
    fixed_poly = poly if poly.is_valid else poly.buffer(0)
    # Sometimes the fixed_poly is still a MultiPolygon,
    # so we need to find the convex hull of the MultiPolygon, which should
    # always be a Polygon (but could be empty).
    if not isinstance(fixed_poly, Polygon):
        fixed_poly = fixed_poly.convex_hull
    return fixed_poly

def rescale_polygon(polygon: ArrayLike,
                    scale_factor: Tuple[int, int],
                    mode: str = 'mul') -> np.ndarray:
    """Rescale a polygon according to scale_factor.

    The behavior is different depending on the mode. When mode is 'mul', the
    coordinates will be multiplied by scale_factor, which is usually used in
    preprocessing transforms such as :func:`Resize`.
    The coordinates will be divided by scale_factor if mode is 'div'. It can be
    used in postprocessors to recover the polygon in the original
    image size.

    Args:
        polygon (ArrayLike): A polygon. In any form can be converted
            to an 1-D numpy array. E.g. list[float], np.ndarray,
            or torch.Tensor. Polygon is written in
            [x1, y1, x2, y2, ...].
        scale_factor (tuple(int, int)): (w_scale, h_scale).
        model (str): Rescale mode. Can be 'mul' or 'div'. Defaults to 'mul'.

    Returns:
        np.ndarray: Rescaled polygon.
    """
    assert len(polygon) % 2 == 0
    assert mode in ['mul', 'div']
    polygon = np.array(polygon, dtype=np.float32)
    poly_shape = polygon.shape
    reshape_polygon = polygon.reshape(-1, 2)
    scale_factor = np.array(scale_factor, dtype=float)
    if mode == 'div':
        scale_factor = 1 / scale_factor
    polygon = (reshape_polygon * scale_factor[None]).reshape(poly_shape)
    return polygon


def valid_boundary(x, with_score=True):
    num = len(x)
    if num < 8:
        return False
    if num % 2 == 0 and (not with_score):
        return True
    if num % 2 == 1 and with_score:
        return True

    return False

def poly_intersection(poly_a: Polygon,
                      poly_b: Polygon,
                      invalid_ret: Optional[Union[float, int]] = None,
                      return_poly: bool = False
                      ) -> Tuple[float, Optional[Polygon]]:
    """Calculate the intersection area between two polygons.

    Args:
        poly_a (Polygon): Polygon a.
        poly_b (Polygon): Polygon b.
        invalid_ret (float or int, optional): The return value when the
            invalid polygon exists. If it is not specified, the function
            allows the computation to proceed with invalid polygons by
            cleaning the their self-touching or self-crossing parts.
            Defaults to None.
        return_poly (bool): Whether to return the polygon of the intersection
            Defaults to False.

    Returns:
        float or tuple(float, Polygon): Returns the intersection area or
        a tuple ``(area, Optional[poly_obj])``, where the `area` is the
        intersection area between two polygons and `poly_obj` is The Polygon
        object of the intersection area, which will be `None` if the input is
        invalid. `poly_obj` will be returned only if `return_poly` is `True`.
    """
    assert isinstance(poly_a, Polygon)
    assert isinstance(poly_b, Polygon)
    assert invalid_ret is None or isinstance(invalid_ret, (float, int))

    if invalid_ret is None:
        poly_a = poly_make_valid(poly_a)
        poly_b = poly_make_valid(poly_b)

    poly_obj = None
    area = invalid_ret
    if poly_a.is_valid and poly_b.is_valid:
        if poly_a.intersects(poly_b):
            poly_obj = poly_a.intersection(poly_b)
            area = poly_obj.area
        else:
            poly_obj = Polygon()
            area = 0.0
    return (area, poly_obj) if return_poly else area


def poly_union(
    poly_a: Polygon,
    poly_b: Polygon,
    invalid_ret: Optional[Union[float, int]] = None,
    return_poly: bool = False
) -> Tuple[float, Optional[Union[Polygon, MultiPolygon]]]:
    """Calculate the union area between two polygons.

    Args:
        poly_a (Polygon): Polygon a.
        poly_b (Polygon): Polygon b.
        invalid_ret (float or int, optional): The return value when the
            invalid polygon exists. If it is not specified, the function
            allows the computation to proceed with invalid polygons by
            cleaning the their self-touching or self-crossing parts.
            Defaults to False.
        return_poly (bool): Whether to return the polygon of the union.
            Defaults to False.

    Returns:
        tuple: Returns a tuple ``(area, Optional[poly_obj])``, where
        the `area` is the union between two polygons and `poly_obj` is the
        Polygon or MultiPolygon object of the union of the inputs. The type
        of object depends on whether they intersect or not. Set as `None`
        if the input is invalid. `poly_obj` will be returned only if
        `return_poly` is `True`.
    """
    assert isinstance(poly_a, Polygon)
    assert isinstance(poly_b, Polygon)
    assert invalid_ret is None or isinstance(invalid_ret, (float, int))

    if invalid_ret is None:
        poly_a = poly_make_valid(poly_a)
        poly_b = poly_make_valid(poly_b)

    poly_obj = None
    area = invalid_ret
    if poly_a.is_valid and poly_b.is_valid:
        poly_obj = poly_a.union(poly_b)
        area = poly_obj.area
    return (area, poly_obj) if return_poly else area

def rescale_polygons(polygons: Union[ArrayLike, Sequence[ArrayLike]],
                     scale_factor: Tuple[int, int],
                     mode: str = 'mul'
                     ) -> Union[ArrayLike, Sequence[np.ndarray]]:
    """Rescale polygons according to scale_factor.

    The behavior is different depending on the mode. When mode is 'mul', the
    coordinates will be multiplied by scale_factor, which is usually used in
    preprocessing transforms such as :func:`Resize`.
    The coordinates will be divided by scale_factor if mode is 'div'. It can be
    used in postprocessors to recover the polygon in the original
    image size.

    Args:
        polygons (list[ArrayLike] or ArrayLike): A list of polygons, each
            written in [x1, y1, x2, y2, ...] and in any form can be converted
            to an 1-D numpy array. E.g. list[list[float]],
            list[np.ndarray], or list[torch.Tensor].
        scale_factor (tuple(int, int)): (w_scale, h_scale).
        model (str): Rescale mode. Can be 'mul' or 'div'. Defaults to 'mul'.

    Returns:
        list[np.ndarray] or np.ndarray: Rescaled polygons. The type of the
        return value depends on the type of the input polygons.
    """
    results = []
    for polygon in polygons:
        results.append(rescale_polygon(polygon, scale_factor, mode))
    if isinstance(polygons, np.ndarray):
        results = np.array(results)
    return results

def poly_iou(poly_a: Polygon,
             poly_b: Polygon,
             zero_division: float = 0.) -> float:
    """Calculate the IOU between two polygons.

    Args:
        poly_a (Polygon): Polygon a.
        poly_b (Polygon): Polygon b.
        zero_division (float): The return value when invalid polygon exists.

    Returns:
        float: The IoU between two polygons.
    """
    assert isinstance(poly_a, Polygon)
    assert isinstance(poly_b, Polygon)
    area_inters = poly_intersection(poly_a, poly_b)
    area_union = poly_union(poly_a, poly_b)
    return area_inters / area_union if area_union != 0 else zero_division
def boundary_iou(src: List,
                 target: List,
                 zero_division: Union[int, float] = 0) -> float:
    """Calculate the IOU between two boundaries.

    Args:
       src (list): Source boundary.
       target (list): Target boundary.
       zero_division (int or float): The return value when invalid
                                    boundary exists.

    Returns:
       float: The iou between two boundaries.
    """
    assert valid_boundary(src, False)
    assert valid_boundary(target, False)
    src_poly = poly2shapely(src)
    target_poly = poly2shapely(target)

    return poly_iou(src_poly, target_poly, zero_division=zero_division)