SOURCE DATA PACKAGE
===================

Scale-Selective Control by Precursor Architecture and Thermal History Enables
Hierarchical Design of Sodium-Ion Layered Oxide Cathodes

This package contains the numerical data plotted in every panel of Figures 1-6 of
the main text and Figures S1-S8 of the Supporting Information.

NAMING
------
Each file is named SourceData_<Figure><panel>.csv, so SourceData_Fig4b.csv holds
the data plotted in Figure 4(b) and SourceData_FigS3-2.csv is the second of the
files behind Figure S3. The mapping is also given in Tables S18 and S19 of the
Supporting Information.

Note that the "repository_file" column of MANIFEST.csv records the name each file
carries inside the analysis repository. A few of those names follow an earlier
figure order and do not match the current panel letter; the names in this package
are the authoritative ones.

CONTENTS
--------
MANIFEST.csv   one row per file: figure, panel, content, row and column counts,
               byte size and SHA-256 checksum. Rows with an empty file name are
               panels whose data is not yet available.
*.csv          the data files, byte-for-byte copies of the frozen analysis
               products. Encoding is UTF-8; the first row is the header.

PROVENANCE AND CALIBER
----------------------
All files were produced by versioned analysis scripts from a single master data
table. They are processed analysis inputs, not raw instrument output.

Two conventions recur and should not be conflated:

  * Particle-size descriptors from scanning electron microscopy (Dsec, D10, D90,
    Span) are number-weighted statistics of two-dimensional projected particles.
    Dsec is the median of the pooled particle set, not an interpolated cumulative
    percentile.
  * Particle-size descriptors of the precursor powders are volume-weighted laser
    diffraction values.

The two are reported separately throughout and are never compared numerically.

Apparent coherent-domain size is a Scherrer estimate under a Gaussian
instrumental-broadening approximation and is used for relative comparison only.
The apparent free-surface line intercept is a two-dimensional descriptor of
surface coarsening and is neither a three-dimensional grain size nor equivalent
to the diffraction quantity.

NOT YET AVAILABLE
-----------------
The electrochemical measurements shown in Figure 6 are quoted in the main text
but have not yet been registered as a frozen data product; those rows appear in
MANIFEST.csv without a file name.

Figure 5(a) is withheld from this package. The analysis repository's manifest
for this panel lists 18 candidate 500x source images, but the panel as
published appears to use a different set of 6 images at 2000x magnification.
This discrepancy was not resolved at packaging time; see MANIFEST.csv for the
placeholder row. The correct source image list should be supplied by the
authors before this panel's data is considered complete.
