from setuptools import setup, find_packages
setup(
    name='blackbox',
    version='1.4.3',
    description='image processing sofware specifically written for the reduction of BlackGEM and MeerLICHT images',
    url='https://github.com/pmvreeswijk/BlackBOX',
    author='Paul Vreeswijk, Kerry Paterson, Danielle Pieterse',
    author_email='pmvreeswijk@gmail.com',
    python_requires='>=3.6',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        #'numpy', 'astropy', 'matplotlib', 'scipy', 'pyfftw',
        #'lmfit', 'sip_tpv', 'scikit-image',
        # dependencies above: zogy
        # dependencies below: blackbox
        'watchdog', 'astroscrappy', 'acstools', 'ephem', 'memory-profiler',
        'astroplan', 'google-cloud-storage', 'google-cloud-pubsub', 'Pillow',
        'google-cloud-logging']
)
