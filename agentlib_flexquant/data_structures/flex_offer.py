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
        description="Power profile of the baseline MPC",
        json_schema_extra={
            "unit": "kW",
            "scalar": False
        }
    )
    pos_price: Optional[float] = pydantic.Field(
        default=None,
        description="Price for positive flexibility",
        json_schema_extra={
            "unit": "ct",
            "scalar": True
        }
    )
    pos_corrected_costs_rel: Optional[float] = pydantic.Field(
        default=None,
        description="Price for positive flexibility, corrected for relative price changes",
        json_schema_extra={
            "unit": "ct/kWh",
            "scalar": True,
        }
    )
    pos_diff_profile: pd.Series = pydantic.Field(
        default=None,
        description="Power profile for the positive difference",
        json_schema_extra={
            "unit": "kW",
            "scalar": False
        }
    )
    neg_price: Optional[float] = pydantic.Field(
        default=None,
        description="Price for negative flexibility",
        json_schema_extra={
            "unit": "ct",
            "scalar": True
        }
    )
    neg_corrected_costs_rel: Optional[float] = pydantic.Field(
        default=None,
        description="Price for negative flexibility, corrected for relative price changes",
        json_schema_extra={
            "unit": "ct/kWh",
            "scalar": True
        }
    )
    neg_diff_profile: pd.Series = pydantic.Field(
        default=None,
        description="Power profile for the negative difference",
        json_schema_extra={
            "unit": "kW",
            "scalar": False
        }
    )
    status: OfferStatus = pydantic.Field(
        default=OfferStatus.NOT_ACCEPTED,
        description="Status of the FlexOffer",
        json_schema_extra={
            "scalar": True
        }
    )
    pos_energy_envelope: Optional[pd.Series] = pydantic.Field(
        default=None,
        description="Positive energy envelope which is the integrated positive power",
        json_schema_extra={
            "scalar": False,
            "unit": "kWh"
        }
    )
    neg_energy_envelope: Optional[pd.Series] = pydantic.Field(
        default=None,
        description="Negative energy envelope which is the integrated negative power",
        json_schema_extra={
            "scalar": False,
            "unit": "kWh"
        }
    )
    base_energy_envelope: Optional[pd.Series] = pydantic.Field(
        default=None,
        description="Base energy envelope which is the integrated base power",
        json_schema_extra={
            "scalar": False,
            "unit": "kWh"
        }
    )
    min_power_envelope: Optional[pd.Series] = pydantic.Field(
        default=None,
        description="Minimum power including inflexible loads in each step of the envelope",
        json_schema_extra={
            "scalar": False,
            "unit": "kW"
        }
    )
    max_power_envelope: Optional[pd.Series] = pydantic.Field(
        default=None,
        description="Maximum power including inflexible loads in each step of the envelope",
        json_schema_extra={
            "scalar": False,
            "unit": "kW"
        }
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
