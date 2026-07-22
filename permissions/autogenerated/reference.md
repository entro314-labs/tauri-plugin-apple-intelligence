## Default Permission

Default permissions for the Apple Intelligence plugin: availability checks,
generation, streaming (incl. cancel), and capability queries (context window, token counting,
supported languages, prewarm). All commands talk only to the local FoundationModels framework —
nothing leaves the device except Private Cloud Compute requests, which are private-by-design.

#### This default permission set includes the following:

- `allow-check-availability`
- `allow-pcc-check-availability`
- `allow-generate`
- `allow-stream`
- `allow-cancel-stream`
- `allow-context-info`
- `allow-token-count`
- `allow-supported-languages`
- `allow-prewarm`

## Permission Table

<table>
<tr>
<th>Identifier</th>
<th>Description</th>
</tr>


<tr>
<td>

`apple-intelligence:allow-cancel-stream`

</td>
<td>

Enables the cancel_stream command without any pre-configured scope.

</td>
</tr>

<tr>
<td>

`apple-intelligence:deny-cancel-stream`

</td>
<td>

Denies the cancel_stream command without any pre-configured scope.

</td>
</tr>

<tr>
<td>

`apple-intelligence:allow-check-availability`

</td>
<td>

Enables the check_availability command without any pre-configured scope.

</td>
</tr>

<tr>
<td>

`apple-intelligence:deny-check-availability`

</td>
<td>

Denies the check_availability command without any pre-configured scope.

</td>
</tr>

<tr>
<td>

`apple-intelligence:allow-context-info`

</td>
<td>

Enables the context_info command without any pre-configured scope.

</td>
</tr>

<tr>
<td>

`apple-intelligence:deny-context-info`

</td>
<td>

Denies the context_info command without any pre-configured scope.

</td>
</tr>

<tr>
<td>

`apple-intelligence:allow-generate`

</td>
<td>

Enables the generate command without any pre-configured scope.

</td>
</tr>

<tr>
<td>

`apple-intelligence:deny-generate`

</td>
<td>

Denies the generate command without any pre-configured scope.

</td>
</tr>

<tr>
<td>

`apple-intelligence:allow-pcc-check-availability`

</td>
<td>

Enables the pcc_check_availability command without any pre-configured scope.

</td>
</tr>

<tr>
<td>

`apple-intelligence:deny-pcc-check-availability`

</td>
<td>

Denies the pcc_check_availability command without any pre-configured scope.

</td>
</tr>

<tr>
<td>

`apple-intelligence:allow-prewarm`

</td>
<td>

Enables the prewarm command without any pre-configured scope.

</td>
</tr>

<tr>
<td>

`apple-intelligence:deny-prewarm`

</td>
<td>

Denies the prewarm command without any pre-configured scope.

</td>
</tr>

<tr>
<td>

`apple-intelligence:allow-stream`

</td>
<td>

Enables the stream command without any pre-configured scope.

</td>
</tr>

<tr>
<td>

`apple-intelligence:deny-stream`

</td>
<td>

Denies the stream command without any pre-configured scope.

</td>
</tr>

<tr>
<td>

`apple-intelligence:allow-supported-languages`

</td>
<td>

Enables the supported_languages command without any pre-configured scope.

</td>
</tr>

<tr>
<td>

`apple-intelligence:deny-supported-languages`

</td>
<td>

Denies the supported_languages command without any pre-configured scope.

</td>
</tr>

<tr>
<td>

`apple-intelligence:allow-token-count`

</td>
<td>

Enables the token_count command without any pre-configured scope.

</td>
</tr>

<tr>
<td>

`apple-intelligence:deny-token-count`

</td>
<td>

Denies the token_count command without any pre-configured scope.

</td>
</tr>
</table>
