import os

import logging
import pytest

import cxflow.utils.download as download

_URL_ZIP = 'https://github.com/Cognexa/cxflow-examples/releases/download/example-files/cxflow-0.12.0.zip'
_URL_ZIP_BASE = os.path.basename(_URL_ZIP)
_URL_RAR = 'https://github.com/Cognexa/cxflow-examples/releases/download/example-files/anomalousTrafficTest.rar'
_URL_RAR_BASE = os.path.basename(_URL_RAR)
_URL_NONE = 'https://github.com/Cognexa/cxflow-examples/releases/download/example-files/worldwide-companies-r10k'
_URL_NONE_BASE = os.path.basename(_URL_NONE)
_URL_INVALID = 'not.valid'
_URL_TO_SANITIZE = 'https://n%0 ?0.=_b//'
_URL_RAISE_ERROR = '://*%~/'

SUCCESS = [_URL_ZIP,        # existing url
           _URL_ZIP + '/']  # existing url ending with slash

UNPACK_FAILURE = [(_URL_RAR, _URL_RAR_BASE, 'anomalousTrafficTest.txt'),    # incorrect type of archive,
                  (_URL_NONE, _URL_NONE_BASE, 'address.csv')]               # filetype not specified

DOWNLOAD_FAILURE = [(_URL_RAR + 'r', _URL_RAR_BASE + 'r'),  # mistyped url
                    (_URL_INVALID, _URL_INVALID)]           # invalid url format


@pytest.mark.parametrize('url', SUCCESS)
def test_download_and_unpack_successful(url, tmpdir, caplog):
    """Check file was downloaded and unpacked."""
    caplog.set_level(logging.INFO)
    download.maybe_download_and_extract(tmpdir, url)

    assert caplog.record_tuples == [('root', logging.INFO, '\tdownloading ' + os.path.join(tmpdir, _URL_ZIP_BASE))]
    assert os.path.exists(os.path.join(tmpdir, _URL_ZIP_BASE))
    assert os.path.exists(os.path.join(tmpdir, 'cxflow-0.12.0/setup.py'))


@pytest.mark.parametrize('url, url_base, path', UNPACK_FAILURE)
def test_download_successful_unpack_impossible(url, url_base, path, tmpdir, caplog):
    """Check file was downloaded but could not be unpacked."""
    caplog.set_level(logging.INFO)
    download.maybe_download_and_extract(tmpdir, url)

    assert 'could not be extracted' in caplog.text
    assert os.path.exists(os.path.join(tmpdir, url_base))
    assert not os.path.exists(os.path.join(tmpdir, path))


@pytest.mark.parametrize('url, url_base', DOWNLOAD_FAILURE)
def test_download_fails(url, url_base, tmpdir, caplog):
    """Given URL is invalid, internet connection is down, etc."""
    caplog.set_level(logging.INFO)
    download.maybe_download_and_extract(tmpdir, url)

    assert 'could not be downloaded' in caplog.text
    assert not os.path.exists(os.path.join(tmpdir, url_base))


def test_url_sanitized(tmpdir, caplog):
    """Only valid characters used to create filename."""
    filename = download.sanitize_url(_URL_TO_SANITIZE)
    assert filename == 'n00._b'

    caplog.set_level(logging.INFO)
    download.maybe_download_and_extract(tmpdir, _URL_TO_SANITIZE)
    assert 'downloading ' + os.path.join(tmpdir, 'n00._b') in caplog.text


def test_url_raise_error(tmpdir):
    """ValueError is raised if there is no valid character to be used as filename."""
    with pytest.raises(ValueError):
        download.sanitize_url(_URL_RAISE_ERROR)

    with pytest.raises(ValueError):
        download.maybe_download_and_extract(tmpdir, _URL_RAISE_ERROR)


def test_filepath_alredy_exists(tmpdir, caplog):
    """File was already downloaded."""
    os.makedirs(os.path.join(tmpdir, _URL_RAR_BASE))

    caplog.set_level(logging.INFO)
    download.maybe_download_and_extract(tmpdir, _URL_RAR)

    assert 'already exists; skipping' in caplog.text
    assert 'downloading' not in caplog.text
