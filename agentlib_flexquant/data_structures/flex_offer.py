"""
Data models for flexibility offers.

This module defines data structures for representing flexibility offers in energy markets,
including baseline power profiles, positive and negative flexibility pricing, and offer
status tracking. The FlexOffer class encapsulates all information needed to represent
a flexibility bid, including power differences from baseline and acceptance status.
"""
from enum import Enum
from typing import Optional

import pandas as pd
import pydantic
from agentlib.core.datamodels import _TYPE_MAP
from pydantic import BaseModel


class OfferStatus(Enum):
    """Status of the FlexOffer"""

    NOT_ACCEPTED = "Not Accepted"
    ACCEPTED_POSITIVE = "Accepted Positive"
    ACCEPTED_NEGATIVE = "Accepted Negative"


class FlexOffer(BaseModel):
    """Data class for the flexibility offer."""

    base_power_profile: pd.Series = pydantic.Field(
        default=None,
        unit="kW",
        scalar=False,
        description="Power profile of the baseline MPC",
    )
    pos_price: Optional[float] = pydantic.Field(
        default=None,
        unit="ct",
        scalar=True,
        description="Price for positive flexibility",
    )
    pos_corrected_costs_rel: Optional[float] = pydantic.Field(
        default=None,
        unit="ct/kWh",
        scalar=True,
        description="Price for positive flexibility, corrected for relative price changes",
    )
    pos_diff_profile: pd.Series = pydantic.Field(
        default=None,
        unit="kW",
        scalar=False,
        description="Power profile for the positive difference",
    )
    neg_price: Optional[float] = pydantic.Field(
        default=None,
        unit="ct",
        scalar=True,
        description="Price for negative flexibility",
    )
    neg_corrected_costs_rel: Optional[float] = pydantic.Field(
        default=None,
        unit="ct/kWh",
        scalar=True,
        description="Price for negative flexibility, corrected for relative price changes",
    )
    neg_diff_profile: pd.Series = pydantic.Field(
        default=None,
        unit="kW",
        scalar=False,
        description="Power profile for the negative difference",
    )
    status: OfferStatus = pydantic.Field(
        default=OfferStatus.NOT_ACCEPTED.value,
        scalar=True,
        description="Status of the FlexOffer",
    )

    class Config:
        """Allow arbitrary (non-Pydantic) types such as pandas.Series or numpy.ndarray
        in model fields without requiring custom validators."""

        arbitrary_types_allowed = True

    def as_dataframe(self) -> pd.DataFrame:
        """Store the flexibility offer in a pd.DataFrame

        Returns:
            DataFrame containing the flexibility offer.
            Scalar values are written on the first timestep.

        """
        data = []
        cols = []

        # append scalar values
        for name, field in self.model_fields.items():
            if field.json_schema_extra["scalar"]:
                ser = pd.Series(getattr(self, name))
                ser.index += self.base_power_profile.index[0]
                data.append(ser)
                cols.append(name)

        df = pd.DataFrame(data).T
        df.columns = cols
        return df


# add the offer type to agent variables
_TYPE_MAP["FlexOffer"] = FlexOffer
