import tempfile
import os.path as op
import sys
import os
try:
    from urllib import pathname2url
except ImportError:
    from urllib.request import pathname2url

import numpy.testing as npt
from nibabel.tmpdirs import TemporaryDirectory
import dipy.data.fetcher as fetcher
from dipy.data import SPHERE_FILES
from threading import Thread
if sys.version_info[0] < 3:
    from SimpleHTTPServer import SimpleHTTPRequestHandler  # Python 2
    from SocketServer import TCPServer as HTTPServer
else:
    from http.server import HTTPServer, SimpleHTTPRequestHandler  # Python 3


def test_check_md5():
    fd, fname = tempfile.mkstemp()
    stored_md5 = fetcher._get_file_md5(fname)
    # If all is well, this shouldn't return anything:
    npt.assert_equal(fetcher.check_md5(fname, stored_md5), None)
    # If None is provided as input, it should silently not check either:
    npt.assert_equal(fetcher.check_md5(fname, None), None)
    # Otherwise, it will raise its exception class:
    npt.assert_raises(fetcher.FetcherError, fetcher.check_md5, fname, 'foo')


def test_make_fetcher():
    symmetric362 = SPHERE_FILES['symmetric362']
    with TemporaryDirectory() as tmpdir:
        stored_md5 = fetcher._get_file_md5(symmetric362)

        # create local HTTP Server
        testfile_url = pathname2url(op.split(symmetric362)[0] + op.sep)
        test_server_url = "http://127.0.0.1:8000/"
        print(testfile_url)
        print(symmetric362)
        current_dir = os.getcwd()
        # change pwd to directory containing testfile.
        os.chdir(testfile_url)
        server = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
        server_thread = Thread(target=server.serve_forever)
        server_thread.deamon = True
        server_thread.start()

        # test make_fetcher
        sphere_fetcher = fetcher._make_fetcher("sphere_fetcher",
                                               tmpdir, test_server_url,
                                               [op.split(symmetric362)[-1]],
                                               ["sphere_name"],
                                               md5_list=[stored_md5])

        sphere_fetcher()
        assert op.isfile(op.join(tmpdir, "sphere_name"))
        npt.assert_equal(fetcher._get_file_md5(op.join(tmpdir, "sphere_name")),
                         stored_md5)

        # stop local HTTP Server
        server.shutdown()
        # change to original working directory
        os.chdir(current_dir)


def test_fetch_data():
    symmetric362 = SPHERE_FILES['symmetric362']
    with TemporaryDirectory() as tmpdir:
        md5 = fetcher._get_file_md5(symmetric362)
        bad_md5 = '8' * len(md5)

        newfile = op.join(tmpdir, "testfile.txt")
        # Test that the fetcher can get a file
        testfile_url = pathname2url(symmetric362)
        print(testfile_url)
        testfile_dir, testfile_name = op.split(testfile_url)
        # create local HTTP Server
        test_server_url = "http://127.0.0.1:8001/" + testfile_name
        current_dir = os.getcwd()
        # change pwd to directory containing testfile.
        os.chdir(testfile_dir)
        # use different port as shutdown() takes time to release socket.
        server = HTTPServer(('localhost', 8001), SimpleHTTPRequestHandler)
        server_thread = Thread(target=server.serve_forever)
        server_thread.deamon = True
        server_thread.start()

        files = {"testfile.txt": (test_server_url, md5)}
        fetcher.fetch_data(files, tmpdir)
        npt.assert_(op.exists(newfile))

        # Test that the file is replaced when the md5 doesn't match
        with open(newfile, 'a') as f:
            f.write("some junk")
        fetcher.fetch_data(files, tmpdir)
        npt.assert_(op.exists(newfile))
        npt.assert_equal(fetcher._get_file_md5(newfile), md5)

        # Test that an error is raised when the md5 checksum of the download
        # file does not match the expected value
        files = {"testfile.txt": (test_server_url, bad_md5)}
        npt.assert_raises(fetcher.FetcherError,
                          fetcher.fetch_data, files, tmpdir)

        # stop local HTTP Server
        server.shutdown()
        # change to original working directory
        os.chdir(current_dir)
