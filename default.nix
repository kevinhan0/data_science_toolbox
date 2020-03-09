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
    # Add packages from nix-env -qaP | grep -i needle queries
    bat
    curl
    exa
    git
    tmux
    wget

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
	geopandas
	notedown
	tensorflowWithCuda
      ];
    })

    # Vim
  ];

  # Customizable development shell setup with at last SSL certs set
  shellHook = ''
  '';
}
