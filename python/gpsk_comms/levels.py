"""Feature ladder for the command link.

The hardened link is six mechanisms stacked on one another, and when the whole
stack is switched on at once a failure anywhere in it looks identical from the
outside: no commands arrive. This module turns the stack into a ladder of
numbered levels so it can be brought up one rung at a time, and so a failure can
be attributed to the exact layer that introduced it.

Each level is the previous level plus exactly one mechanism:

===== ============ =========================================================
Level Name         Adds
===== ============ =========================================================
0     ``basic``    nothing -- a bare point-to-point BPSK link
1     ``fec``      convolutional coding and interleaving
2     ``auth``     keyed frame authentication and replay rejection
3     ``dsss``     keyed spreading code (processing gain)
4     ``excision`` narrowband interference excision
5     ``hop``      keyed sub-channel frequency hopping
6     ``escape``   LO retuning and band escape
===== ============ =========================================================

Level 0 is deliberately as close to nothing as the framing allows: differential
BPSK bursts, a public constant sync word, no key of any kind. If level 0 does not
pass traffic the problem is antennas, gain, sample rate or clocks -- not
anti-jam. Working upwards, the first level that fails is the layer at fault.

Everything above the rung being tested is off, and everything below stays on, so
each rung is a genuine superset of the one under it. That is the property worth
preserving when adding a mechanism: put it at the top rather than folding it into
an existing one.

Ordering rationale
------------------
The order is by increasing cost of being wrong, not purely by implementation
complexity. Two constraints are load-bearing rather than aesthetic:

* **Excision must sit above spreading.** Excision deletes spectral bins that
  stand well above the median, which is precisely what an *unspread* signal
  looks like. Enabled without spreading it deletes the wanted signal and nothing
  else, so the dependency is enforced below rather than merely recommended.
* **Hopping must sit above spreading**, because the per-slot spreading code is
  what makes one hop slot distinguishable from another.

Authentication is placed below spreading because it is the only rung that is
purely arithmetic: it cannot be broken by anything on the radio, so a failure
there is unambiguous. Spreading is where RF reality re-enters.
"""

#: Levels, in the order they are built up. Index is the level number.
LEVEL_NAMES = ("basic", "fec", "auth", "dsss", "excision", "hop", "escape")

LEVEL_BASIC = 0
LEVEL_FEC = 1
LEVEL_AUTH = 2
LEVEL_DSSS = 3
LEVEL_EXCISION = 4
LEVEL_HOP = 5
LEVEL_ESCAPE = 6

#: The default for both ends: everything except band escape, which needs a radio
#: and a schedule both ends agreed in advance (see :mod:`gpsk_comms.hopping`).
DEFAULT_LEVEL = LEVEL_HOP

#: The single mechanism each level adds. ``None`` at level 0, which adds nothing.
#: Building the feature set by accumulating this mapping is what guarantees every
#: level is a superset of its predecessor -- there is no way to express a level
#: that drops something a lower one had.
LEVEL_ADDS = {
    LEVEL_BASIC: None,
    LEVEL_FEC: "fec",
    LEVEL_AUTH: "auth",
    LEVEL_DSSS: "dsss",
    LEVEL_EXCISION: "excision",
    LEVEL_HOP: "hopping",
    LEVEL_ESCAPE: "band_escape",
}

#: Feature flags, in ladder order.
FEATURES = ("fec", "auth", "dsss", "excision", "hopping", "band_escape")

#: What each feature does, for :meth:`LinkProfile.describe` and the diagnostics.
FEATURE_SUMMARY = {
    "fec": "rate-1/2 K=7 convolutional coding with block interleaving",
    "auth": "keyed HMAC frame authentication and replay rejection",
    "dsss": "keyed spreading code, for processing gain against a broadband jammer",
    "excision": "frequency-domain removal of narrowband jammers",
    "hopping": "keyed sub-channel hopping within the sampled window",
    "band_escape": "LO retuning and escape to the other allocation",
}

#: Features that cannot be enabled on their own. These are correctness
#: constraints, not style: see the module docstring for why each one exists.
FEATURE_REQUIRES = {
    "excision": "dsss",
    "hopping": "dsss",
    "band_escape": "hopping",
}

#: Features that need key material. ``auth`` needs the MAC key; ``dsss`` and
#: ``hopping`` need the spreading and hop keys even when frames are not
#: authenticated, which is why this is broader than ``auth`` alone.
KEYED_FEATURES = ("auth", "dsss", "hopping", "band_escape")


class LinkProfile:
    """An immutable set of enabled features, plus the level it came from.

    Read the flags directly (``profile.dsss``); build one with :func:`profile`.
    """

    __slots__ = FEATURES + ("level", "name")

    def __init__(self, level, name, **features):
        self.level = level
        self.name = name
        for feature in FEATURES:
            setattr(self, feature, bool(features.get(feature, False)))

    @property
    def enabled(self):
        """Enabled feature names, in ladder order."""
        return tuple(feature for feature in FEATURES if getattr(self, feature))

    @property
    def needs_key(self):
        """True when this profile cannot run without a master key.

        Level 0 and level 1 need no key at all. That is the point of them: the
        entire "wrong key at one end" class of failure is removed from the
        bring-up before the radio has been shown to work.
        """
        return any(getattr(self, feature) for feature in KEYED_FEATURES)

    def describe(self):
        """Multi-line summary, used by the diagnostics and the flowgraphs."""
        lines = [f"level {self.level} ({self.name})"]
        for feature in FEATURES:
            mark = "on " if getattr(self, feature) else "off"
            lines.append(f"  [{mark}] {feature:<12s} {FEATURE_SUMMARY[feature]}")
        if not self.needs_key:
            lines.append("  no key required at this level")
        return "\n".join(lines)

    def __repr__(self):
        return f"LinkProfile(level={self.level}, {', '.join(self.enabled) or 'nothing'})"

    def __eq__(self, other):
        if not isinstance(other, LinkProfile):
            return NotImplemented
        return self.level == other.level and all(
            getattr(self, feature) == getattr(other, feature) for feature in FEATURES
        )

    def __hash__(self):
        return hash((self.level,) + tuple(getattr(self, f) for f in FEATURES))


def normalise_level(level):
    """Accept a level number or a level name and return the number."""
    if isinstance(level, LinkProfile):
        return level.level
    if isinstance(level, str):
        text = level.strip().lower()
        if text.isdigit():
            return normalise_level(int(text))
        try:
            return LEVEL_NAMES.index(text)
        except ValueError:
            raise ValueError(
                f"unknown level name {level!r}; expected one of {', '.join(LEVEL_NAMES)}"
            ) from None
    number = int(level)
    if not 0 <= number < len(LEVEL_NAMES):
        raise ValueError(f"level must be between 0 and {len(LEVEL_NAMES) - 1}; got {level}")
    return number


def features_for_level(level):
    """Return the cumulative feature set at ``level`` as a dict."""
    number = normalise_level(level)
    enabled = {feature: False for feature in FEATURES}
    for rung in range(number + 1):
        added = LEVEL_ADDS[rung]
        if added is not None:
            enabled[added] = True
    return enabled


def profile(level=DEFAULT_LEVEL, **overrides):
    """Build a :class:`LinkProfile` for ``level``, with optional overrides.

    ``overrides`` takes any feature name from :data:`FEATURES`. ``None`` means
    "leave the level's own setting alone", which is what lets a caller pass an
    unset option straight through without having to know the default.

    Overrides exist for bisecting a fault rather than for routine configuration:
    turning one mechanism off inside an otherwise complete stack is how you
    confirm which one is responsible. Combinations that cannot work -- excision
    without spreading, for instance -- are rejected rather than silently
    producing a dead link.
    """
    number = normalise_level(level)
    enabled = features_for_level(number)

    for name, value in overrides.items():
        if name not in enabled:
            raise ValueError(
                f"unknown feature {name!r}; expected one of {', '.join(FEATURES)}"
            )
        if value is not None:
            enabled[name] = bool(value)

    for feature, requirement in FEATURE_REQUIRES.items():
        if enabled[feature] and not enabled[requirement]:
            raise ValueError(
                f"{feature!r} requires {requirement!r}: {_why_required(feature)}"
            )

    return LinkProfile(number, LEVEL_NAMES[number], **enabled)


def _why_required(feature):
    if feature == "excision":
        return (
            "excision blanks spectral bins that stand above the median, which is "
            "exactly what an unspread signal looks like -- it would delete the "
            "wanted signal and nothing else"
        )
    if feature == "hopping":
        return "the per-slot spreading code is what distinguishes one hop slot from another"
    return "band escape moves the whole hop schedule to another allocation"


def ladder():
    """Return every level as a list of profiles, lowest first.

    The diagnostics walk this in order and stop at the first rung that fails.
    """
    return [profile(number) for number in range(len(LEVEL_NAMES))]
