# Built-in Functions

481 functions, read from the code that dispatches them. Every name links to a page of its own.

## Arrays

- [array_concat](array-concat.md), The arrays joined end to end.
- [array_contains](array-contains.md), True when the array holds an element equal to the value.
- [array_distinct](array-distinct.md), The array with duplicates removed, keeping first-appearance order.
- [array_filter](array-filter.md), The elements the lambda holds true for, in order.
- [array_length](array-length.md), The number of elements an array holds.
- [array_position](array-position.md), The 1-based position of the first element equal to a value, or NULL when it holds none.
- [array_slice](array-slice.md), The elements from a 1-based start position, up to a length.
- [array_sort](array-sort.md), The array sorted ascending, nulls last.
- [array_to_string](array-to-string.md), The elements rendered and joined by a delimiter.
- [array_transform](array-transform.md), Each element replaced by the lambda's value for it.
- [string_to_array](string-to-array.md), The text split on a delimiter into an array of elements.

## Bitfields and hashing

- [adler32](adler32.md), Adler-32 checksum, faster than CRC-32 and weaker on short inputs.
- [base32_decode](base32-decode.md), Base32 text back to bytes.
- [base58_decode](base58-decode.md), Base58 text back to bytes.
- [base64url_decode](base64url-decode.md), URL-safe base64 text back to bytes.
- [bitfield_all](bitfield-all.md), Whether every bit of a mask is set in a field.
- [bitfield_and](bitfield-and.md), Bits set in both fields.
- [bitfield_any](bitfield-any.md), Whether at least one bit of a mask is set in a field.
- [bitfield_clear](bitfield-clear.md), Turns one bit off.
- [bitfield_count](bitfield-count.md), How many bits are set.
- [bitfield_from_positions](bitfield-from-positions.md), A field with the listed bits set.
- [bitfield_not](bitfield-not.md), Every bit inverted.
- [bitfield_or](bitfield-or.md), Bits set in either field.
- [bitfield_set](bitfield-set.md), Turns one bit on.
- [bitfield_test](bitfield-test.md), Whether one bit is set.
- [bitfield_to_positions](bitfield-to-positions.md), The positions of the set bits, lowest first.
- [bitfield_toggle](bitfield-toggle.md), Flips one bit.
- [bitfield_xor](bitfield-xor.md), Bits set in one field but not both.
- [blake3](blake3.md), BLAKE3 digest, 32 bytes.
- [city_hash](city-hash.md), Alias for cityhash64.
- [cityhash64](cityhash64.md), 64-bit CityHash.
- [consistent_hash](consistent-hash.md), Bucket for a key, stable as the bucket count changes.
- [fnv1a_64](fnv1a-64.md), 64-bit FNV-1a hash, simple and fast on short inputs.
- [fnvhash](fnvhash.md), Alias for fnv1a_64.
- [hash_combine](hash-combine.md), Folds two hashes into one.
- [hex_decode](hex-decode.md), Hexadecimal text back to bytes.
- [hmac_sha256](hmac-sha256.md), Keyed SHA-256 authentication code, 32 bytes.
- [murmur3_128](murmur3-128.md), 128-bit MurmurHash3.
- [murmur3_32](murmur3-32.md), 32-bit MurmurHash3.
- [sha256](sha256.md), SHA-256 digest, 32 bytes.
- [sha384](sha384.md), SHA-384 digest, 48 bytes.
- [sha512](sha512.md), SHA-512 digest, 64 bytes.
- [siphash](siphash.md), SipHash, a keyed hash that resists collision flooding.
- [xxhash32](xxhash32.md), 32-bit xxHash.

## Colour and fingerprinting

- [address_similarity](address-similarity.md), Similarity of two street addresses, normalising abbreviations.
- [color_blend](color-blend.md), Mixes two colours in a given proportion.
- [color_darken](color-darken.md), Lowers a colour's lightness.
- [color_from_hex](color-from-hex.md), Reads a hex colour string into a packed colour.
- [color_from_hsl](color-from-hsl.md), Builds a colour from hue, saturation and lightness.
- [color_from_rgb](color-from-rgb.md), Packs three channel values into a colour.
- [color_from_rgba](color-from-rgba.md), Packs three channel values and an alpha into a colour.
- [color_hex](color-hex.md), Alias for color_from_hex.
- [color_hsl](color-hsl.md), Alias for color_from_hsl.
- [color_lighten](color-lighten.md), Raises a colour's lightness.
- [color_palette](color-palette.md), Colours related to a base colour by a scheme.
- [color_rgb](color-rgb.md), Alias for color_from_rgb.
- [color_rgba](color-rgba.md), Alias for color_from_rgba.
- [color_to_hex](color-to-hex.md), Writes a colour as a hex string.
- [color_to_hsl](color-to-hsl.md), Reads a colour as hue, saturation and lightness.
- [company_similarity](company-similarity.md), Similarity of two company names, ignoring legal suffixes.
- [cosine_similarity](cosine-similarity.md), Cosine of the angle between two numeric vectors.
- [double_metaphone](double-metaphone.md), Two phonetic codes for a word, covering alternate pronunciations.
- [hamming](hamming.md), Number of positions at which two equal-length strings differ.
- [jaccard_similarity](jaccard-similarity.md), Share of the combined token set that both sets hold.
- [minhash_decode](minhash-decode.md), Reads a MinHash signature back as numbers.
- [minhash_encode](minhash-encode.md), Writes a MinHash signature as bytes.
- [minhash_signature](minhash-signature.md), MinHash signature over a token set.
- [minhash_similarity](minhash-similarity.md), Jaccard similarity estimated from two MinHash signatures.
- [name_similarity](name-similarity.md), Similarity of two personal names, allowing for nicknames.
- [ngram_similarity](ngram-similarity.md), Jaccard similarity over two strings' character n-grams.
- [overlap_coefficient](overlap-coefficient.md), Shared token count over the smaller set's size.
- [qgram_distance](qgram-distance.md), Difference between two strings' q-gram counts.
- [shingle](shingle.md), Overlapping runs of k characters from a string.
- [simhash](simhash.md), 64-bit fingerprint of a document's word content.
- [simhash_distance](simhash-distance.md), Number of differing bits between two SimHash fingerprints.
- [simhash_similar](simhash-similar.md), Whether two SimHash fingerprints are within a bit distance.
- [sorensen_dice](sorensen-dice.md), Twice the shared token count over the two set sizes.
- [wcag_compliant](wcag-compliant.md), Whether two colours meet a WCAG contrast level.
- [wcag_contrast_ratio](wcag-contrast-ratio.md), Contrast ratio between two colours under WCAG 2.0.
- [word_shingle](word-shingle.md), Overlapping runs of k words from a string.

## Data quality metrics

- [collated_compare](collated-compare.md), Compares two strings under a named collation.
- [collation_sort_key](collation-sort-key.md), A byte key whose plain ordering is a locale's ordering.
- [distinct_rate](distinct-rate.md), Passes when a column's distinct share reaches a threshold.
- [freshness](freshness.md), Passes when the newest timestamp in a batch is recent enough.
- [map_value](map-value.md), One entry of a MAP, by key.
- [nested_json](nested-json.md), A stored nested value rendered as JSON.
- [null_rate](null-rate.md), Passes when a column's NULL share stays at or below a threshold.
- [struct_field](struct-field.md), One declared field of a STRUCT, by position.
- [validate_credit_card](validate-credit-card.md), Whether a card number passes the Luhn check.
- [validate_ean](validate-ean.md), Whether text is a valid EAN-8, EAN-13 or UPC-A barcode number.
- [validate_email](validate-email.md), Whether text is a well-formed email address.
- [validate_iban](validate-iban.md), Whether text is a valid IBAN.
- [validate_isbn](validate-isbn.md), Whether text is a valid ISBN-10 or ISBN-13.
- [validate_issn](validate-issn.md), Whether text is a valid ISSN.
- [validate_json](validate-json.md), Whether text is syntactically valid JSON.
- [validate_ssn](validate-ssn.md), Whether text is a well-formed US Social Security number.
- [validate_swift](validate-swift.md), Whether text is a valid SWIFT or BIC code.
- [validate_url](validate-url.md), Whether text parses as a URL.
- [validate_uuid](validate-uuid.md), Whether text is a hyphenated UUID.
- [validate_vin](validate-vin.md), Whether text is a valid vehicle identification number.
- [variant_extract](variant-extract.md), The part of a VARIANT a path names.

## Diff and patch

- [row_diff](row-diff.md), Which columns of a row changed, matched by name.
- [text_diff](text-diff.md), Line-level difference between two strings.
- [text_patch](text-patch.md), Applies a patch to a string.

## Finance and time series

- [amortization_schedule](amortization-schedule.md), Payment, interest, principal and balance for every period of a loan.
- [bond_price](bond-price.md), Present value of a bond's coupons and face value.
- [bond_yield](bond-yield.md), Yield implied by a bond's price.
- [compound_interest](compound-interest.md), Value of a principal compounded a number of times a year.
- [depreciation_db](depreciation-db.md), Declining balance depreciation for one period.
- [depreciation_sl](depreciation-sl.md), Straight-line depreciation per period.
- [depreciation_syd](depreciation-syd.md), Sum-of-years-digits depreciation for one period.
- [fv](fv.md), Value of an investment after a number of periods.
- [irr](irr.md), Rate at which evenly spaced cashflows break even.
- [lttb](lttb.md), Downsamples a series to a point count, keeping its shape.
- [npv](npv.md), Net present value of evenly spaced cashflows.
- [pmt](pmt.md), Periodic payment that repays a loan.
- [pv](pv.md), Value today of a future stream of payments.
- [time_bucket_calendar](time-bucket-calendar.md), Start of the calendar bucket a timestamp falls in.
- [time_bucket_gapfill_calendar](time-bucket-gapfill-calendar.md), Every calendar bucket boundary across a range.
- [xirr](xirr.md), Annualised rate at which dated cashflows break even.
- [xnpv](xnpv.md), Net present value of cashflows on given dates.

## Fuzzy text matching

- [damerau_levenshtein](damerau-levenshtein.md), Edit distance that counts a transposition of adjacent characters as one edit.
- [jaro_similarity](jaro-similarity.md), Similarity weighted by matching characters and their transpositions.
- [jaro_winkler](jaro-winkler.md), Jaro similarity that scores a shared prefix higher.
- [levenshtein](levenshtein.md), Edit distance between two strings, counting insertions, deletions and substitutions.
- [levenshtein_similarity](levenshtein-similarity.md), Edit distance expressed as a similarity between 0 and 1.
- [metaphone](metaphone.md), Phonetic code with more consonant detail than soundex and no fixed length.
- [nysiis](nysiis.md), Phonetic code tuned for surnames, including non-English ones.
- [soundex](soundex.md), Four-character phonetic code, so names that sound alike compare equal.

## Identifiers and generated keys

- [bayesian_average](bayesian-average.md), Rating pulled toward a global average by its vote count.
- [cuid2](cuid2.md), Alias for gen_cuid2.
- [elo_expected](elo-expected.md), Expected win probability for the first of two Elo ratings.
- [elo_update](elo-update.md), New Elo rating after one result.
- [gen_cuid2](gen-cuid2.md), A collision-resistant identifier beginning with a letter.
- [gen_ksuid](gen-ksuid.md), A time-ordered 27-character identifier with 128 random bits.
- [gen_nanoid](gen-nanoid.md), A URL-safe random identifier of a chosen length.
- [gen_snowflake](gen-snowflake.md), A 64-bit identifier holding a timestamp, a machine number and a sequence.
- [gen_tsid](gen-tsid.md), A 64-bit identifier holding a timestamp and random bits.
- [gen_ulid](gen-ulid.md), A time-ordered 26-character identifier.
- [gen_uuid_v4](gen-uuid-v4.md), A random UUID.
- [gen_uuid_v7](gen-uuid-v7.md), A time-ordered UUID.
- [glicko2_update](glicko2-update.md), New Glicko-2 rating, deviation and volatility after a rating period.
- [iban_bban](iban-bban.md), Domestic account part of an IBAN.
- [iban_country](iban-country.md), Country code of an IBAN.
- [isbn_format](isbn-format.md), Writes an ISBN with hyphens in the 10 or 13 digit form.
- [isbn_to_13](isbn-to-13.md), Converts a 10-digit ISBN to its 13-digit form.
- [ksuid](ksuid.md), Alias for gen_ksuid.
- [nanoid](nanoid.md), A URL-safe random identifier of a chosen length.
- [semver_format](semver-format.md), Writes a packed version back as text.
- [semver_increment_major](semver-increment-major.md), Next major version.
- [semver_increment_minor](semver-increment-minor.md), Next minor version.
- [semver_increment_patch](semver-increment-patch.md), Next patch version.
- [semver_is_prerelease](semver-is-prerelease.md), Whether a packed version is a pre-release.
- [semver_major](semver-major.md), Major component of a packed version.
- [semver_minor](semver-minor.md), Minor component of a packed version.
- [semver_parse](semver-parse.md), Packs a semantic version into one sortable value.
- [semver_patch](semver-patch.md), Patch component of a packed version.
- [semver_satisfies](semver-satisfies.md), Whether a version meets a constraint.
- [snowflake](snowflake.md), Alias for gen_snowflake.
- [trueskill_update](trueskill-update.md), New TrueSkill means and deviations after a match.
- [tsid](tsid.md), Alias for gen_tsid.
- [ulid](ulid.md), Alias for gen_ulid.
- [uuid_to_string](uuid-to-string.md), Writes UUID bytes in the hyphenated form.
- [uuid_v4](uuid-v4.md), Alias for gen_uuid_v4.
- [uuid_v7](uuid-v7.md), Alias for gen_uuid_v7.
- [vin_country](vin-country.md), Country or region a vehicle was built in.
- [vin_manufacturer](vin-manufacturer.md), World manufacturer identifier from a VIN.
- [vin_year](vin-year.md), Model year from a VIN.
- [wilson_score](wilson-score.md), Lower bound of the Wilson confidence interval on a proportion.
- [win_rate](win-rate.md), Wins as a share of games played.

## JSON and VARIANT

- [json_diff](json-diff.md), A patch turning one JSON value into another.
- [json_get](json-get.md), A member or element of a JSON value, as JSON.
- [json_get_path](json-get-path.md), A nested part of a JSON value, as JSON.
- [json_get_path_text](json-get-path-text.md), A nested part of a JSON value, as text.
- [json_get_text](json-get-text.md), A member or element of a JSON value, as text.
- [json_merge_patch](json-merge-patch.md), Merges a JSON document into another.
- [json_patch](json-patch.md), Applies a patch to a JSON value.
- [jsonb_contained_by](jsonb-contained-by.md), Whether one JSON value is held by another.
- [jsonb_contains](jsonb-contains.md), Whether one JSON value holds another.
- [jsonb_exists](jsonb-exists.md), Whether a JSON object holds a member.
- [jsonb_exists_all](jsonb-exists-all.md), Whether a JSON object holds all of several members.
- [jsonb_exists_any](jsonb-exists-any.md), Whether a JSON object holds any of several members.

## Label trees

- [build_path](build-path.md), A path from a list of labels.
- [lca](lca.md), Longest ancestor every path shares.
- [ltree_index](ltree-index.md), Where a run of labels appears in a path.
- [ltree_is_ancestor](ltree-is-ancestor.md), Whether the first path is at or above the second.
- [ltree_is_descendant](ltree-is-descendant.md), Whether the first path is at or below the second.
- [ltree_matches](ltree-matches.md), Whether a path matches a label query.
- [ltree_matches_any](ltree-matches-any.md), Whether a path matches any of several label queries.
- [nlevel](nlevel.md), How many labels a path holds.
- [subpath](subpath.md), Part of a path, by label position.

## Masking

- [masking_email](masking-email.md), An address with the local part hashed and the domain kept.
- [masking_ip](masking-ip.md), An address with its host part zeroed.
- [masking_name](masking-name.md), A name reduced to its initials.
- [masking_phone](masking-phone.md), A number with its digits starred out.
- [masking_ssn](masking-ssn.md), A number reduced to a hash and its last four digits.

## Matrices and geometry

- [cross_product](cross-product.md), Vector perpendicular to two three-dimensional vectors.
- [dot_product](dot-product.md), Sum of the element-wise products of two vectors.
- [eigenvalues](eigenvalues.md), Eigenvalues of a symmetric matrix, largest first.
- [h3_distance](h3-distance.md), Grid steps between two cells at the same resolution.
- [h3_from_point](h3-from-point.md), Grid cell index covering a point at a resolution.
- [h3_to_boundary](h3-to-boundary.md), Boundary polygon of a grid cell.
- [matrix_add](matrix-add.md), Element-wise sum of two matrices of the same shape.
- [matrix_create](matrix-create.md), Builds a matrix from a row-major array of numbers.
- [matrix_determinant](matrix-determinant.md), Determinant of a square matrix by LU decomposition.
- [matrix_identity](matrix-identity.md), The n by n identity matrix.
- [matrix_inverse](matrix-inverse.md), Inverse of a square matrix by Gauss-Jordan elimination.
- [matrix_multiply](matrix-multiply.md), Matrix product of two conformable matrices.
- [matrix_norm](matrix-norm.md), Size of a matrix under one of four norms.
- [matrix_scalar_multiply](matrix-scalar-multiply.md), Multiplies every element of a matrix by one number.
- [matrix_subtract](matrix-subtract.md), Element-wise difference of two matrices of the same shape.
- [matrix_trace](matrix-trace.md), Sum of the diagonal elements of a square matrix.
- [matrix_transpose](matrix-transpose.md), Reflects a matrix across its diagonal.
- [pca](pca.md), Principal component analysis of a samples by features matrix.
- [st_area](st-area.md), Area of a polygon by the shoelace formula.
- [st_as_geojson](st-as-geojson.md), GeoJSON for a geometry.
- [st_as_text](st-as-text.md), Well-known text for a geometry.
- [st_buffer](st-buffer.md), Polygon approximating a circle around a point.
- [st_centroid](st-centroid.md), Centre of a geometry as a point.
- [st_contains](st-contains.md), Whether a polygon holds a point.
- [st_distance](st-distance.md), Distance between two points.
- [st_dwithin](st-dwithin.md), Whether two points lie within a radius of each other.
- [st_geom_from_geojson](st-geom-from-geojson.md), A geometry from a GeoJSON geometry object.
- [st_geom_from_text](st-geom-from-text.md), A geometry from well-known text.
- [st_intersects](st-intersects.md), Whether two geometries share any position.
- [st_make_point](st-make-point.md), A point geometry from a longitude and a latitude.
- [st_union](st-union.md), Collection holding both geometries.
- [svd](svd.md), Singular value decomposition of a matrix.

## Media

- [audio_metadata](audio-metadata.md), Duration, sample rate and codec facts about an audio payload.
- [audio_transcode](audio-transcode.md), Re-encodes audio at a codec and bitrate.
- [audio_transcribe](audio-transcribe.md), Speech in an audio payload as text.
- [audio_trim](audio-trim.md), The part of an audio payload between two offsets.
- [barcode_decode](barcode-decode.md), The value a linear barcode holds.
- [data_matrix_decode](data-matrix-decode.md), The value a Data Matrix code holds.
- [data_matrix_encode](data-matrix-encode.md), A Data Matrix image for a value.
- [detect_encoding](detect-encoding.md), Character encoding of a text payload.
- [detect_mime_type](detect-mime-type.md), Media type of a payload, from its leading bytes.
- [document_extract_text](document-extract-text.md), Plain text of a document.
- [document_metadata](document-metadata.md), Title, author and other recorded facts about a document.
- [document_page_count](document-page-count.md), Number of pages in a document.
- [document_to_markdown](document-to-markdown.md), Markdown of a document, keeping its headings and lists.
- [file_extension](file-extension.md), Conventional file extension for a media type.
- [html_to_markdown](html-to-markdown.md), Markdown for an HTML fragment.
- [html_to_text](html-to-text.md), Readable text of an HTML fragment.
- [image_crop](image-crop.md), Cuts a rectangle out of an image.
- [image_embed](image-embed.md), Vector embedding of an image.
- [image_format](image-format.md), Re-encodes an image in another format.
- [image_metadata](image-metadata.md), Format, dimensions and other header facts about an image.
- [image_ocr](image-ocr.md), Text recognised in an image.
- [image_resize](image-resize.md), Resizes an image to a width and height.
- [image_rotate](image-rotate.md), Turns an image clockwise.
- [is_binary](is-binary.md), Whether a payload looks like binary rather than text.
- [markdown_to_html](markdown-to-html.md), HTML for a markdown document.
- [presigned_url](presigned-url.md), A signed, expiring link to a stored payload.
- [presigned_verify](presigned-verify.md), Whether a signed link is valid and unexpired.
- [qr_decode](qr-decode.md), The value a QR code holds.
- [qr_encode](qr-encode.md), A QR code image for a value.
- [sanitize_html](sanitize-html.md), HTML with everything but a safe set of tags removed.
- [video_extract_audio](video-extract-audio.md), Audio track of a video, as MP3.
- [video_extract_frame](video-extract-frame.md), One frame of a video as an image.
- [video_metadata](video-metadata.md), Duration, dimensions and codec facts about a video.
- [video_thumbnail](video-thumbnail.md), A representative still from a video.
- [video_transcode](video-transcode.md), Re-encodes a video at a codec, bitrate and resolution.

## Money and quantities

- [convert_units](convert-units.md), A plain number converted between two units.
- [currency_by_numeric](currency-by-numeric.md), A currency's details from its ISO 4217 numeric code.
- [currency_lookup](currency-lookup.md), A currency's details from its ISO 4217 alphabetic code.
- [format_bytes](format-bytes.md), A byte count written with a unit.
- [format_currency](format-currency.md), A number as a currency amount, in a locale's conventions.
- [format_duration](format-duration.md), A number of seconds written as a duration.
- [format_number](format-number.md), A number with locale group and decimal separators.
- [format_ordinal](format-ordinal.md), A number written as an English ordinal.
- [format_percentage](format-percentage.md), A fraction as a percentage with a percent sign.
- [money_add](money-add.md), Sum of two money values of the same currency.
- [money_convert](money-convert.md), A money value restated in another currency at a given rate.
- [money_create](money-create.md), A money value from an amount and a currency code.
- [money_currency_code](money-currency-code.md), The three-letter ISO 4217 code of a money value's currency.
- [money_currency_symbol](money-currency-symbol.md), The display symbol of a money value's currency.
- [money_format](money-format.md), A money value as text with its currency symbol and minor digits.
- [money_minor_digits](money-minor-digits.md), How many decimal places the currency uses.
- [money_multiply](money-multiply.md), A money value scaled by a number.
- [money_subtract](money-subtract.md), Difference of two money values of the same currency.
- [parse_number](parse-number.md), Locale-formatted number text back to a number.
- [quantity_add](quantity-add.md), Sum of two quantities of the same dimension.
- [quantity_convert](quantity-convert.md), A quantity restated in another unit of the same dimension.
- [quantity_create](quantity-create.md), A quantity from a number and a unit of measure.
- [quantity_dimension](quantity-dimension.md), The physical dimension the quantity measures.
- [quantity_format](quantity-format.md), A quantity as text with its unit symbol.
- [quantity_multiply](quantity-multiply.md), A quantity scaled by a number, keeping its unit.
- [quantity_scale](quantity-scale.md), A quantity scaled by a number.
- [quantity_subtract](quantity-subtract.md), Difference of two quantities of the same dimension.
- [quantity_unit_name](quantity-unit-name.md), The full name of the quantity's unit.

## Network and URL

- [cidr_parse](cidr-parse.md), Reads text as a network, refusing host bits outside the prefix.
- [inet_broadcast](inet-broadcast.md), The last address in an address's network, with its host bits set.
- [inet_contains](inet-contains.md), True when the network covers the address.
- [inet_family](inet-family.md), 4 for an IPv4 address, 6 for IPv6.
- [inet_format](inet-format.md), The address as text, including its prefix length when it has one.
- [inet_host](inet-host.md), The address without its prefix length.
- [inet_is_loopback](inet-is-loopback.md), True when the address refers to the local host.
- [inet_is_private](inet-is-private.md), True when the address is in a range reserved for private use.
- [inet_netmask](inet-netmask.md), The prefix length written as a mask address.
- [inet_network](inet-network.md), The network an address sits in, with its host bits cleared.
- [inet_parse](inet-parse.md), Reads text as an IP address, with an optional prefix length.
- [inet_prefix](inet-prefix.md), The prefix length in bits.
- [macaddr_format](macaddr-format.md), The hardware address as colon-separated lower-case text.
- [macaddr_oui](macaddr-oui.md), The first three bytes, which identify the hardware manufacturer.
- [macaddr_parse](macaddr-parse.md), Reads text as a hardware address.
- [url_domain](url-domain.md), The registrable domain, excluding subdomains.
- [url_fragment](url-fragment.md), The fragment, without the hash.
- [url_host](url-host.md), The host, without the port.
- [url_is_absolute](url-is-absolute.md), True when the URL carries a scheme.
- [url_normalize](url-normalize.md), One spelling for URLs that address the same resource.
- [url_parse](url-parse.md), Splits a URL into its parts as a structured value.
- [url_path](url-path.md), The path, excluding the query and the fragment.
- [url_port](url-port.md), The port written in the URL.
- [url_query_param](url-query-param.md), The value of one query parameter, decoded.
- [url_query_params](url-query-params.md), Every query parameter as key and value pairs.
- [url_resolve](url-resolve.md), A relative reference resolved against a base URL.
- [url_scheme](url-scheme.md), The scheme, without the colon.
- [url_tld](url-tld.md), The public suffix the host sits under.

## Probabilistic sketches

- [bloom_contains](bloom-contains.md), Whether a Bloom filter may hold a value.
- [cms_estimate](cms-estimate.md), Estimated count for one value in a Count-Min Sketch.
- [hll_count](hll-count.md), Estimated distinct count held by a HyperLogLog sketch.

## Probability and statistics

- [bloom_add](bloom-add.md), Adds one value to a Bloom filter and returns the filter.
- [bloom_create](bloom-create.md), An empty Bloom filter sized for an item count and a target false positive rate.
- [bloom_false_positive_rate](bloom-false-positive-rate.md), Current false positive rate of a Bloom filter from its fill.
- [bloom_merge](bloom-merge.md), Union of two Bloom filters.
- [cms_add](cms-add.md), Adds to a value's count in a Count-Min Sketch and returns the sketch.
- [cms_create](cms-create.md), An empty Count-Min Sketch of the given width and depth.
- [cms_merge](cms-merge.md), Counter-wise sum of two Count-Min Sketches.
- [correlation](correlation.md), Pearson correlation coefficient between two series.
- [covariance](covariance.md), Sample covariance between two series.
- [exponential_smoothing](exponential-smoothing.md), Smooths a series by weighting each value against the running smoothed value.
- [forecast_linear](forecast-linear.md), Projects a least squares line to new x values.
- [hll_add](hll-add.md), Adds one value to a HyperLogLog sketch and returns the sketch.
- [hll_create](hll-create.md), An empty HyperLogLog sketch at the given precision.
- [hll_error](hll-error.md), Relative standard error of a HyperLogLog sketch.
- [hll_merge](hll-merge.md), Union of two HyperLogLog sketches.
- [kurtosis](kurtosis.md), Tail weight of a series, in excess form.
- [linear_regression](linear-regression.md), Least squares straight line fitted to two series.
- [moving_average](moving-average.md), Trailing simple moving average over a fixed window.
- [outlier_detect_iqr](outlier-detect-iqr.md), Flags values outside Tukey's fences around the interquartile range.
- [outlier_detect_zscore](outlier-detect-zscore.md), Flags values further than a threshold of standard deviations from the mean.
- [percentile](percentile.md), Exact percentile of a series by linear interpolation.
- [skewness](skewness.md), Asymmetry of a series about its mean.
- [stddev_pop](stddev-pop.md), Population standard deviation of a series.
- [stddev_sample](stddev-sample.md), Sample standard deviation of a series.
- [tdigest_add](tdigest-add.md), Adds one value to a T-Digest and returns the digest.
- [tdigest_cdf](tdigest-cdf.md), Fraction of a T-Digest's weight at or below a value.
- [tdigest_create](tdigest-create.md), An empty T-Digest at the given compression.
- [tdigest_merge](tdigest-merge.md), Combines two T-Digests into one distribution.
- [tdigest_quantile](tdigest-quantile.md), Estimated value at a quantile of a T-Digest.
- [variance_pop](variance-pop.md), Population variance of a series.
- [variance_sample](variance-sample.md), Sample variance of a series.
- [weighted_moving_average](weighted-moving-average.md), Moving average with an explicit weight per window position.
- [zscore](zscore.md), How many standard deviations a value sits from a mean.

## Regular expressions and strings

- [camel_case](camel-case.md), Joins words with no separator, capitalising each after the first.
- [extract_emails](extract-emails.md), Email addresses found in free text.
- [extract_phone_numbers](extract-phone-numbers.md), Telephone numbers found in free text.
- [extract_urls](extract-urls.md), URLs found in free text.
- [initcap](initcap.md), Upper-cases the first letter of each word and lower-cases the rest.
- [ip_compare](ip-compare.md), Compares two IP addresses by value.
- [is_valid_date](is-valid-date.md), Whether text is a date in a given layout.
- [json_schema_errors](json-schema-errors.md), Why a JSON value fails a schema.
- [json_schema_validate](json-schema-validate.md), Whether a JSON value satisfies a schema.
- [kebab_case](kebab-case.md), Lower-cases words and joins them with hyphens.
- [natural_compare](natural-compare.md), Compares two strings treating digit runs as numbers.
- [natural_sort_key](natural-sort-key.md), A byte key whose plain ordering is natural ordering.
- [pascal_case](pascal-case.md), Joins words with no separator, capitalising every one.
- [path_compare](path-compare.md), Compares two paths component by component.
- [regex_capture](regex-capture.md), The first match and its capture groups.
- [regex_compile](regex-compile.md), Checks a pattern and returns a value the compiled forms accept.
- [regex_count](regex-count.md), How many times a pattern matches.
- [regex_find](regex-find.md), Position of the first match.
- [regex_find_all](regex-find-all.md), Positions of every match.
- [regex_find_compiled](regex-find-compiled.md), Position of the first match of a validated pattern.
- [regex_match](regex-match.md), Whether a pattern matches anywhere in the text.
- [regex_match_compiled](regex-match-compiled.md), Whether a validated pattern matches anywhere in the text.
- [regex_replace](regex-replace.md), Replaces the first match.
- [regex_replace_all](regex-replace-all.md), Replaces every match.
- [regex_split](regex-split.md), The text cut at every match.
- [slug](slug.md), Text reduced to a lower-case URL segment.
- [snake_case](snake-case.md), Lower-cases words and joins them with underscores.
- [strip_html](strip-html.md), Plain text of an HTML fragment.
- [text_diff_words](text-diff-words.md), Word-level difference between two strings.
- [title_case](title-case.md), Capitalises words as a title, leaving short words lower-case.
- [truncate_chars](truncate-chars.md), Shortens text to a character count and marks the cut.
- [truncate_words](truncate-words.md), Shortens text to a word count.
- [validate_json_schema](validate-json-schema.md), Alias for json_schema_validate.
- [version_compare](version-compare.md), Compares two version strings.

## State machines and rate limits

- [fixed_window_count](fixed-window-count.md), Which fixed window a timestamp falls in.
- [leaky_bucket_add](leaky-bucket-add.md), Adds work to a leaky bucket and reports whether it fit.
- [leaky_bucket_create](leaky-bucket-create.md), An empty leaky bucket.
- [sliding_window_check](sliding-window-check.md), Whether the window still has room below a limit.
- [sliding_window_count](sliding-window-count.md), How many events fall in the window ending now.
- [sm_available_events](sm-available-events.md), Events that can be applied from a state.
- [sm_can_transition](sm-can-transition.md), Whether an event is allowed from a state.
- [sm_is_terminal](sm-is-terminal.md), Whether a state has no way out.
- [sm_parse](sm-parse.md), Reads a state machine definition and returns it in compiled form.
- [sm_reachable_states](sm-reachable-states.md), Every state reachable from a starting state.
- [sm_shortest_path](sm-shortest-path.md), Fewest events that move the machine between two states.
- [sm_transition](sm-transition.md), The state an event moves the machine to.
- [token_bucket_available](token-bucket-available.md), Tokens a bucket holds at a given time.
- [token_bucket_consume](token-bucket-consume.md), Takes tokens from a bucket and reports whether they were there.
- [token_bucket_create](token-bucket-create.md), A full token bucket.

## Text, encoding and markup

- [barcode_encode](barcode-encode.md), A barcode image for a value.
- [base32_encode](base32-encode.md), Bytes as base32 text, using A to Z and 2 to 7.
- [base58_encode](base58-encode.md), Bytes as base58 text, with no characters that look alike.
- [base64url_encode](base64url-encode.md), Bytes as base64 text using the URL-safe alphabet.
- [change_log](change-log.md), Renders a table's recorded changes between two versions.
- [crc32](crc32.md), CRC-32 checksum, for detecting accidental corruption.
- [crc32c](crc32c.md), CRC-32C checksum, the Castagnoli polynomial.
- [cume_dist](cume-dist.md), Share of the partition at or below the current row.
- [custom_order_rank](custom-order-rank.md), Position of a value in a stated order.
- [delta](delta.md), Change from the previous row.
- [dense_rank](dense-rank.md), Rank within the partition, leaving no gaps after a tie.
- [derivative](derivative.md), Change per second, with the time column stated.
- [ema](ema.md), Exponentially weighted moving average over the window.
- [first_value](first-value.md), The value from the first row of the window frame.
- [hex_encode](hex-encode.md), Bytes as lower-case hexadecimal, two characters per byte.
- [ip_sort_key](ip-sort-key.md), A byte key whose plain ordering is numeric address ordering.
- [json_diff_table](json-diff-table.md), Differences between two JSON documents as rows.
- [json_equals](json-equals.md), Whether two JSON documents hold the same value.
- [lag](lag.md), A value from an earlier row of the partition.
- [last_value](last-value.md), The value from the last row of the window frame.
- [lead](lead.md), A value from a later row of the partition.
- [markdown_extract_code_blocks](markdown-extract-code-blocks.md), The fenced code blocks of a markdown document.
- [markdown_extract_headers](markdown-extract-headers.md), The headings of a markdown document with their levels.
- [markdown_extract_links](markdown-extract-links.md), The links of a markdown document with their text.
- [moving_avg](moving-avg.md), Average over the window frame.
- [nth_value](nth-value.md), The value from the nth row of the window frame.
- [ntile](ntile.md), Which of n equal buckets the row falls in.
- [percent_rank](percent-rank.md), Rank as a fraction of the partition, from 0.
- [rank](rank.md), Rank within the partition, leaving gaps after a tie.
- [rate](rate.md), Change per second from the previous row.
- [row_diff_ordinal](row-diff-ordinal.md), Which columns of a row changed, matched by position.
- [row_number](row-number.md), Position of the row within its partition.
- [xxhash128](xxhash128.md), 128-bit xxHash3.
- [xxhash64](xxhash64.md), 64-bit xxHash, for hash tables and partitioning.

## Time, ranges and schedules

- [add_business_days](add-business-days.md), Moves a date forward or back by working days.
- [business_days_between](business-days-between.md), Count of working days in a date span.
- [cron_between](cron-between.md), Every time a cron expression fires in a window.
- [cron_human_readable](cron-human-readable.md), A cron expression written out field by field.
- [cron_list](cron-list.md), The next few times a cron expression fires.
- [cron_matches](cron-matches.md), Whether a cron expression fires at an instant.
- [cron_next](cron-next.md), First time a cron expression fires after an instant.
- [cron_parse](cron-parse.md), Compiles a cron expression to its parsed form.
- [cron_prev](cron-prev.md), Last time a cron expression fired before an instant.
- [day_of_week](day-of-week.md), Day of the week as a number, Sunday first.
- [fiscal_quarter](fiscal-quarter.md), Quarter of the fiscal year holding a date.
- [fiscal_year](fiscal-year.md), Fiscal year a date falls in.
- [is_business_day](is-business-day.md), Whether a date is a working day.
- [next_business_day](next-business-day.md), First working day after a date.
- [parse_natural_date](parse-natural-date.md), Reads a relative date phrase against a reference date.
- [parse_natural_duration](parse-natural-duration.md), Reads a duration phrase as an interval.
- [range_adjacent](range-adjacent.md), Whether two ranges meet with no gap and no overlap.
- [range_contains_range](range-contains-range.md), Whether one range holds another entirely.
- [range_contains_value](range-contains-value.md), Whether a range holds a value.
- [range_create](range-create.md), Builds a range from two bounds and their inclusivity.
- [range_intersection](range-intersection.md), Range of the values two ranges share.
- [range_is_empty](range-is-empty.md), Whether a range holds no values.
- [range_lower](range-lower.md), Encoded lower bound of a range.
- [range_lower_inclusive](range-lower-inclusive.md), Whether a range holds its lower bound.
- [range_overlaps](range-overlaps.md), Whether two ranges share any value.
- [range_union](range-union.md), Single range covering two that touch.
- [range_upper](range-upper.md), Encoded upper bound of a range.
- [range_upper_inclusive](range-upper-inclusive.md), Whether a range holds its upper bound.
- [week_of_fiscal_year](week-of-fiscal-year.md), Week number within the fiscal year.

## Vectors and versions

- [bloom_filter_estimate_count](bloom-filter-estimate-count.md), How many distinct items a Bloom filter probably holds.
- [convert_currency](convert-currency.md), Converts a money value to another currency at a stored rate.
- [money_round](money-round.md), Rounds a money value to a number of decimal places.
- [parse_money](parse-money.md), Reads a formatted money string as a money value.
- [phonetic_match](phonetic-match.md), Whether two strings encode to the same phonetic code.
- [phonetic_score](phonetic-score.md), How close two strings are phonetically, from 0 to 1.
- [semver_compare](semver-compare.md), Compares two version strings by SemVer precedence.
- [semver_prerelease](semver-prerelease.md), The pre-release tag of a version string.
- [semver_sort](semver-sort.md), Version strings ordered by full SemVer precedence.
- [vector_angle](vector-angle.md), Angle between two stored vectors, in radians.
- [vector_cross](vector-cross.md), Cross product of two three-dimensional stored vectors.
- [vector_dot](vector-dot.md), Dot product of two stored vectors.
- [vector_norm](vector-norm.md), Length of a stored vector.
- [vector_normalize](vector-normalize.md), A vector scaled to length 1.

481 have a page. 297 state an argument list.
