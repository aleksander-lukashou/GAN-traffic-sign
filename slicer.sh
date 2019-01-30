#!/bin/bash
for filename in *; do
    slice-image $filename 64
done
