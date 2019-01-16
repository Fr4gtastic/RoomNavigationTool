import multiprocessing
import os
from functools import partial

import flickrapi
from keras.utils.data_utils import urlretrieve

FLICKR_KEY = '7c85c283eeb984827f193929745829b0'
FLICKR_SECRET = 'fee1c56a8ed23859'

flickr = flickrapi.FlickrAPI(FLICKR_KEY, FLICKR_SECRET, format='parsed-json')


def flickr_url(photo, size=''):
    url = 'http://farm{farm}.staticflickr.com/{server}/{id}_{secret}{size}.jpg'
    if size:
        size = '_' + size
    return url.format(size=size, **photo)


def fetch_photo(dir_name, photo):
    urlretrieve(flickr_url(photo), os.path.join(dir_name, photo['id'] + '.jpg'))


def fetch_image_set(query, dir_name, page=1, count=500, sort='relevance'):
    res = flickr.photos.search(text='"{}"'.format(query), per_page=count, sort=sort, page=page)['photos']['photo']
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with multiprocessing.Pool() as p:
        p.map(partial(fetch_photo, dir_name), res)


if __name__ == '__main__':
    page = 0
    for _ in range(10):
        page += 1
        fetch_image_set('room window', dir_name=r'data4\train\window', page=page)
        fetch_image_set('room door', dir_name=r'data4\train\door', page=page)
        fetch_image_set('room', dir_name=r'data4\train\anything_else', page=page)
