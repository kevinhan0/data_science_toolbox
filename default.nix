with import <nixpkgs> {};
stdenv.mkDerivation rec {
  name = "env";

  # Mandatory boilerplate for buildable env
  env = buildEnv { name = name; paths = buildInputs; };
  allowUnfree = true;
  builder = builtins.toFile "builder.sh" ''
    source $stdenv/setup; ln -s $env $out
  '';

  # Customizable development requirements
  buildInputs = [
    # With Python configuration requiring a special wrapper
    (python37.buildEnv.override {
      ignoreCollisions = true;
      extraLibs = with python37Packages; [
        numpy
        matplotlib
        scipy
        ipython
        scikitlearn
        nltk
        Keras
        pandas
        black
        lightgbm
        seaborn
        shapely
        jupyter
        jupyterlab
        pyspark
        xgboost
        #geopandas
        #notedown
        #tensorflowWithCuda
      ];
    })
  ];

  # Customizable development shell setup with at last SSL certs set
  shellHook = ''
  '';
}
